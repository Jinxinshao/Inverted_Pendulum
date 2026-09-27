"""Read-only diagnostics, calibration helpers and the motor step test."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from ..hw.pd42s1 import PD42S1, DriverError, hexs
from ..hw.sensor import AngleCalibration, SensorReader, parse_line, sensor_statistics


def open_serial(port: str, baud: int = 115200, timeout: float = 0.01):
    import serial  # pyserial

    s = serial.Serial(port, baudrate=baud, timeout=timeout, write_timeout=0.1)
    s.reset_input_buffer()
    return s


# ------------------------------------------------------------------ sensor
def record_sensor(port, seconds: float) -> list[tuple[float, int, int | None]]:
    rd = SensorReader(port, keep_seconds=seconds + 5)
    rd.start()
    time.sleep(seconds)
    rd.stop()
    rd.join(timeout=1.0)
    if rd.error:
        raise IOError(rd.error)
    data = rd.snapshot()
    data_stats = {"lines": rd.lines, "parse_errors": rd.parse_errors}
    record_sensor.last_counts = data_stats  # type: ignore[attr-defined]
    return data


def diagnose_sensor(port_name: str, baud: int, seconds: float, calib: AngleCalibration, out=print) -> dict:
    out(f"== angle sensor on {port_name} @ {baud} (read-only, {seconds:.0f} s) ==")
    port = open_serial(port_name, baud)
    try:
        t0 = time.perf_counter()
        raw = b""
        while time.perf_counter() - t0 < 0.3:
            raw += port.read(256)
        lines = [ln for ln in raw.split(b"\n") if ln.strip()][:5]
        out("first raw lines:")
        for ln in lines:
            out(f"   {ln!r}  -> parsed {parse_line(ln)}")
        data = record_sensor(port, seconds)
    finally:
        port.close()
    st = sensor_statistics(data)
    st.update(getattr(record_sensor, "last_counts", {}))
    if st.get("n", 0) > 2:
        st["theta_mean_deg"] = math.degrees(calib.theta(st["adc_mean"]))
        st["theta_noise_std_deg"] = st["adc_std"] * calib.deg_per_count
    for k, v in st.items():
        out(f"   {k:>22s}: {v:.4g}" if isinstance(v, float) else f"   {k:>22s}: {v}")
    out(_sensor_verdict(st))
    return st


def _sensor_verdict(st: dict) -> str:
    msgs = []
    r = st.get("rate_hz", 0)
    if r < 100:
        msgs.append(f"RATE TOO LOW ({r:.0f} Hz): balancing needs >= 200 Hz. Change the ADC firmware (docs/05 step 1).")
    elif r < 190:
        msgs.append(f"rate {r:.0f} Hz: use loop.Ts >= {1.0 / r * 1.05:.4f} s or raise the firmware rate.")
    else:
        msgs.append(f"rate {r:.0f} Hz OK for Ts = 5 ms.")
    if st.get("dt_max_ms", 0) > 20:
        msgs.append(f"gaps up to {st['dt_max_ms']:.0f} ms seen - USB/OS stalls; close other serial tools.")
    if st.get("parse_errors", 0):
        msgs.append(f"{st['parse_errors']} unparsable lines.")
    if st.get("adc_max", 0) >= 4075 or st.get("adc_min", 4095) <= 20:
        msgs.append("ADC at an end of range (potentiometer end/dead zone) during the recording.")
    return "verdict: " + " ".join(msgs)


def save_sensor_csv(data, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    t0 = data[0][0] if data else 0.0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "adc", "mcu_t"])
        for t, a, m in data:
            w.writerow([f"{t - t0:.6f}", a, "" if m is None else m])


def load_sensor_csv(path: str, with_phase: bool = False):
    """t, adc (and the 'phase' column rest/swing written by record_potentiometer.py, '' if absent)."""
    t, a, ph = [], [], []
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            t.append(float(row["t"]))
            a.append(float(row["adc"]))
            ph.append(row.get("phase") or "")
    if with_phase:
        return np.array(t), np.array(a), np.array(ph)
    return np.array(t), np.array(a)


def split_rest_swing(t: np.ndarray, phase: np.ndarray | None = None, rest_s: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (rest, swing). Priority: phase column; a pause > 1 s (the recorder
    waits for Enter between the two phases); else the first ``rest_s`` seconds."""
    if phase is not None and np.any(phase == "rest") and np.any(phase == "swing"):
        return phase == "rest", phase == "swing"
    gaps = np.diff(t)
    if len(gaps) and gaps.max() > 1.0:
        k = int(np.argmax(gaps)) + 1
        idx = np.arange(len(t))
        return idx < k, idx >= k
    rest = t < t[0] + rest_s
    return rest, ~rest


def swing_angle_from_adc(adc: np.ndarray, calib: AngleCalibration, lo: int = 20, hi: int = 4075) -> np.ndarray:
    """Angle from the BOTTOM (rad) for free-swing identification; NaN in the dead zone."""
    th = calib.sign * (adc - calib.adc_upright) / calib.counts_per_rad
    phi = (th % (2 * math.pi)) - math.pi  # theta = pi at the bottom -> phi = 0
    phi = np.where((adc < lo) | (adc > hi), np.nan, phi)
    return phi


def average_adc(port, seconds: float) -> tuple[float, float, int]:
    data = record_sensor(port, seconds)
    a = np.array([d[1] for d in data], float)
    return float(np.mean(a)), float(np.std(a)), len(a)


def calibrate_angle_interactive(port_name: str, baud: int, out=print, ask=input) -> dict:
    """Guided procedure; returns the 'hardware.calibration' dict."""
    port = open_serial(port_name, baud)
    try:
        out("Step A - hanging reference. Let the rod hang freely and come to rest.")
        out("         Tap it lightly from the LEFT, wait until still, press Enter.")
        ask()
        d1, s1, n1 = average_adc(port, 2.0)
        out(f"         ADC = {d1:.1f} (std {s1:.2f}, n={n1})")
        out("         Now tap it lightly from the RIGHT, wait until still, press Enter.")
        ask()
        d2, s2, n2 = average_adc(port, 2.0)
        out(f"         ADC = {d2:.1f} (std {s2:.2f}, n={n2})")
        down = 0.5 * (d1 + d2)
        out(f"         hanging ADC = {down:.1f}; friction band = +-{abs(d1 - d2) / 2:.1f} counts")
        if down > 4075 or down < 20:
            out("         NOTE: hanging position is at the potentiometer end of track (expected on this rig).")
        out("Step B - upright. Hold the rod as vertical as you can (use a spirit level / plumb line")
        out("         against the rod), keep still, press Enter.")
        ask()
        up, su, nu = average_adc(port, 2.0)
        out(f"         upright ADC = {up:.1f} (std {su:.2f}, n={nu})")
        cpr_meas = abs(down - up) / math.pi
        cpr_sheet = 4095 / math.radians(345.0)
        out(f"         slope from gravity reference: {cpr_meas:.1f} counts/rad; datasheet: {cpr_sheet:.1f} counts/rad")
        if down > 4075 or down < 20:
            # the hanging reading is clipped at the track end: not a valid 180 deg reference
            cpr_meas = AngleCalibration().counts_per_rad
            out(f"         hanging reading is clipped at the track end -> using the free-swing slope {cpr_meas:.1f} "
                "(identify --csv gives the value for your rod)")
        out("Step C - sign. Tilt the rod ~10 deg TOWARD the end of the rail the cart moves to for a")
        out("         POSITIVE speed command (docs/05 step 4 defines it; before that: toward the motor), press Enter.")
        ask()
        tilt, _, _ = average_adc(port, 1.0)
        sign = 1 if tilt > up else -1
        out(f"         ADC {tilt:.0f} vs upright {up:.0f} -> sign = {sign:+d}")
    finally:
        port.close()
    return {
        "adc_upright": round(up, 2),
        "counts_per_rad": round(cpr_meas, 2),
        "sign": sign,
        "_adc_hanging": round(down, 2),
        "_friction_band_counts": round(abs(d1 - d2) / 2, 2),
        "_note": "adc_upright is refined later by the Kalman bias estimate (see docs/05 step 6)",
    }


# ------------------------------------------------------------------ motor
def diagnose_motor(port_name: str, baud: int, address: int, protocol: str, seconds: float = 2.0, out=print) -> dict:
    """Read-only: version, system, drive and homing parameters, plus latency of a read."""
    out(f"== PD42S1 on {port_name} @ {baud}, address {address}, protocol {protocol} (read-only) ==")
    port = open_serial(port_name, baud)
    res: dict = {}
    try:
        t0 = time.perf_counter()
        idle = bytearray()
        while time.perf_counter() - t0 < min(seconds, 1.0):
            idle += port.read(256)
        out(f"passive listen: {len(idle)} unsolicited bytes" + (f": {hexs(bytes(idle[:48]))}" if idle else ""))
        drv = PD42S1(port, protocol, address, timeout=0.1)
        try:
            fw, hw_ = drv.read_version()
            out(f"firmware V{fw:.1f}, hardware V{hw_:.1f}")
        except DriverError as e:
            out(f"read version failed: {e}")
            out("-> check: 12 V on? port? baud? protocol (custom/modbus) as set in the official software? address?")
            return {"ok": False, "error": str(e)}
        res["system"] = drv.read_system()
        res["drive"] = drv.read_drive()
        res["homing"] = drv.read_homing()
        for sec in ("system", "drive", "homing"):
            out(f"-- {sec}")
            for kk, v in res[sec].items():
                out(f"   {kk:>20s}: {v:.4g}" if isinstance(v, float) else f"   {kk:>20s}: {v}")
        lat = []
        for _ in range(20):
            t1 = time.perf_counter()
            drv.read_position()
            lat.append(time.perf_counter() - t1)
        out(f"read_position round trip: median {np.median(lat) * 1e3:.2f} ms, max {np.max(lat) * 1e3:.2f} ms")
        ok, problems, _ = drv.readiness()
        out("ready for closed loop" if ok else "NOT ready: " + "; ".join(problems))
        res.update(ok=True, ready=ok, problems=problems, latency_ms=[x * 1e3 for x in lat])
    finally:
        port.close()
    return res


def probe_motor(port_name: str, baud: int, address: int, protocol: str, rate_hz: float, count: int,
                csv_path: str | None, out=print) -> dict:
    """Repeated read-only transactions (real-time position 0x2A) with error classification."""
    port = open_serial(port_name, baud)
    drv = PD42S1(port, protocol, address, timeout=min(0.1, 0.9 / rate_hz))
    rows = []
    stats = {"ok": 0, "timeout": 0, "garbled": 0, "driver_error": 0}
    try:
        period = 1.0 / rate_hz
        t_start = time.perf_counter()
        for k in range(count):
            target = t_start + k * period
            while time.perf_counter() < target:
                time.sleep(0.0005)
            t0 = time.perf_counter()
            try:
                pos = drv.read_position()
                err, lat = "ok", (time.perf_counter() - t0) * 1e3
            except DriverError as e:
                msg = str(e)
                err = "timeout" if "timeout" in msg else ("garbled" if "garbled" in msg else "driver_error")
                pos, lat = "", ""
            stats[err] += 1
            rec = drv.records[-1]
            rows.append((f"{t0 - t_start:.4f}", hexs(rec.tx), hexs(rec.rx), lat, pos, err))
    finally:
        port.close()
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "tx", "rx", "latency_ms", "position", "result"])
            w.writerows(rows)
    lats = [r[3] for r in rows if r[3] != ""]
    out(f"{count} reads at {rate_hz} Hz: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    if lats:
        out(f"latency ms: median {np.median(lats):.2f}, p99 {np.percentile(lats, 99):.2f}, max {np.max(lats):.2f}")
    stats["error_rate"] = 1.0 - stats["ok"] / max(1, count)
    return stats


def motor_step_test(drv: PD42S1, units, v_step: float = 0.05, t_move: float = 0.4, accel: int = 0, out=print) -> dict:
    """Velocity step with fast position polling; fit x(t) = g v (s - tau (1 - e^{-s/tau})), s = t - L.

    Moves the cart by about v_step * t_move (default 2 cm). Needs calibrated units.
    """
    from scipy.optimize import least_squares

    pts = []
    x0 = units.x_of_counts(drv.read_position())
    t0 = time.perf_counter()
    drv.set_speed(units.rpm_of(v_step), accel)
    try:
        while time.perf_counter() - t0 < t_move:
            t1 = time.perf_counter()
            c = drv.read_position()
            pts.append((0.5 * (t1 + time.perf_counter()) - t0, units.x_of_counts(c) - x0))
    finally:
        drv.set_speed(0.0, accel)
    t = np.array([p[0] for p in pts])
    x = np.array([p[1] for p in pts])

    def model(par):
        L, tau, g = par
        s_ = np.clip(t - L, 0, None)
        return g * v_step * (s_ - tau * (1 - np.exp(-s_ / max(tau, 1e-4))))

    fit = least_squares(lambda par: model(par) - x, [0.01, 0.02, 1.0], bounds=([0, 1e-4, 0.2], [0.2, 0.5, 5.0]))
    L, tau, g = fit.x
    res = {"latency_s": float(L), "tau_s": float(tau), "gain": float(g), "n_points": len(t),
           "rms_fit_mm": float(np.sqrt(np.mean(fit.fun**2)) * 1e3), "t": t.tolist(), "x": x.tolist()}
    out(f"step {v_step:+.3f} m/s: latency {L * 1e3:.1f} ms, lag tau {tau * 1e3:.1f} ms, speed gain {g:.3f} "
        f"(1.0 = calibration correct), fit rms {res['rms_fit_mm']:.2f} mm, {len(t)} samples")
    return res


def update_config_file(path: str, section: list[str], values: dict) -> None:
    p = Path(path)
    cfg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d = cfg
    for k in section:
        d = d.setdefault(k, {})
    d.update(values)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def cart_calibrate_interactive(drv: PD42S1, rpm: float = 20.0, t_move: float = 1.5, out=print, ask=input) -> dict:
    """Measure belt travel per motor revolution and define +x.

    +x is DEFINED as the direction the cart moves for a positive (正转) speed, so
    sign = +1; the encoder direction is checked. Result: meters_per_rev.
    """
    out("1) Put the cart at the MIDDLE of the rail, mark its position with tape, keep hands clear, press Enter.")
    ask()
    c0 = drv.read_position()
    out(f"   encoder = {c0}")
    out(f"2) Moving with +{rpm} rpm (正转) for {t_move:.1f} s ...")
    drv.set_speed(rpm, 0)
    time.sleep(t_move)
    drv.set_speed(0.0, 0)
    time.sleep(0.3)
    c1 = drv.read_position()
    turns = (c1 - c0) / 51200.0
    out(f"   encoder = {c1}  ({turns:+.4f} motor turns)")
    ans = ask("3) Measure the distance from the tape mark to the cart's new position in millimetres: ")
    dx = float(ans) / 1000.0
    if abs(turns) < 1e-3:
        raise RuntimeError("encoder did not change: is the motor enabled and in speed mode?")
    mpr = dx / abs(turns)
    enc_sign = 1 if turns > 0 else -1
    out(f"   belt travel per motor turn = {mpr * 1000:.2f} mm/rev (GT2 20T would be 40.0 mm)")
    out(f"   encoder direction for +rpm: {'+' if enc_sign > 0 else '-'}  -> +x = direction the cart just moved; MARK IT")
    out("4) Move the cart back to the tape mark (e.g. jog in the 驱动器 tab), then press Enter to store the centre.")
    ask()
    cc = drv.read_position()
    return {"meters_per_rev": round(mpr, 6), "center_counts": cc, "sign": enc_sign}
