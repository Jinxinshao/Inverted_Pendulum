"""Command line interface:  python -m pendulum_lab <command> [options]

Run ``python -m pendulum_lab -h`` or ``python -m pendulum_lab <command> -h``.
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

from . import __version__
from .config import ALGORITHMS, load_config, make_controller, make_estimator, plant_from, sim_config_from, swingup_overrides


def _cfg(args) -> dict:
    return load_config(getattr(args, "config", None))


def _print_dict(d: dict, indent: str = "   ") -> None:
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{indent}{k}: {v:.6g}")
        elif isinstance(v, dict):
            print(f"{indent}{k}:")
            _print_dict(v, indent + "   ")
        else:
            print(f"{indent}{k}: {v}")


# ------------------------------------------------------------------ model
def cmd_params(args) -> None:
    from .control.analysis import rule_of_thumb_delay
    from .model.identification import fit_decay_summary_both
    from .model.params import PendulumGeometry

    cfg = _cfg(args)
    p = plant_from(cfg)
    print("== geometry estimate (steel tube 8x1 mm, 50 cm, pivot 12 cm from bottom, Al head 3 cm) ==")
    _print_dict(PendulumGeometry().mass_properties())
    print("== stopwatch free-swing test: 20 cycles, 90 -> 50 deg, 24.85 s ==")
    for f in fit_decay_summary_both():
        print("   " + f.summary())
    print("== plant parameters in use ==")
    _print_dict(p.to_dict())
    print(f"   consistent (omega0^2 = m g l / J within 5 %): {p.consistent()}")
    print(f"   rule-of-thumb latency limit 0.5/p = {rule_of_thumb_delay(p) * 1e3:.0f} ms")


def identify_csv(path: str, cfg: dict, min_amp: float = 5.0, out=print) -> dict:
    """Free-swing CSV -> both identification methods + comparison. Returns dict of fits."""
    import numpy as np

    from .control.online_id import ls_free_swing
    from .model.identification import extrema, fit_free_decay, fit_free_decay_one_sided
    from .tools.hwtools import load_sensor_csv, swing_angle_from_adc

    cal = _calib_or_default(cfg)
    t, adc = load_sensor_csv(path)
    rest = adc[t < 2.5]
    if len(rest) > 10 and np.std(rest) < 3:
        down = float(np.median(rest))
        out(f"rest (hanging) ADC = {down:.1f} -> counts/rad from gravity reference = {abs(down - cal.adc_upright) / np.pi:.1f}")
    phi = swing_angle_from_adc(adc, cal)
    dead = np.isnan(phi).mean()
    out(f"{len(t)} samples, {len(t) / (t[-1] - t[0]):.1f} Hz, dead-zone/invalid samples {dead * 100:.1f} %")
    te, ve = extrema(t, phi, min_amp=np.radians(min_amp))
    # which side never wraps around the resistive track? its extremum readings lie
    # between the upright and the hanging reading (no jump through the dead zone)
    idx = np.searchsorted(t, te).clip(0, len(adc) - 1)
    lo, hi = sorted((cal.adc_upright, float(np.nanmedian(adc[t < 2.5])) if (t < 2.5).any() else 4092.0))
    inside = (adc[idx] >= lo - 50) & (adc[idx] <= hi + 50)
    side = int(np.sign(np.sum(np.sign(ve[inside])))) or 1
    out(f"non-wrapped side: {'+' if side > 0 else '-'} ({inside.sum()} of {len(te)} extrema read without wrap)")
    fits: dict = {"side": side}
    fits["one_sided"] = fo = fit_free_decay_one_sided(t, phi, side, min_amp_deg=min_amp)
    out("method A1 (same-side extrema, full cycles, energy balance)  [recommended]: " + fo.summary())
    try:
        fits["two_sided"] = fe = fit_free_decay(t, phi, min_amp_deg=min_amp)
        out("method A2 (both sides, half cycles)  [biased by the wrap offset on this rig]: " + fe.summary())
    except ValueError as e:
        out(f"method A2 failed: {e}")
    segs_phi = np.where(np.sign(phi) == side, phi, np.nan)  # same-side samples only (no wrap offset)
    try:
        fits["svf_ls"] = fl = ls_free_swing(t, segs_phi, t_start=float(te[0]) if len(te) else None)
        out("method B (state-variable filter + least squares on the ODE, same side): " + fl.summary())
    except ValueError as e:
        out(f"method B failed: {e}")
    return fits


def cmd_identify(args) -> None:
    from .model.identification import fit_decay_summary_both
    from .tools.hwtools import update_config_file

    if args.csv:
        cfg = _cfg(args)
        fits = identify_csv(args.csv, cfg, args.min_amp)
        fe = fits["one_sided"]
        if args.write:
            update_config_file(args.write, ["plant"], {"omega0": round(fe.omega0, 5), "viscous_c": round(fe.viscous_c, 5),
                                                       "coulomb_gamma": round(fe.coulomb_gamma, 5), "source": f"free decay {args.csv}"})
            print(f"plant parameters (method A) written to {args.write}")
    else:
        a0, a1, n, T = args.summary
        for f in fit_decay_summary_both(a0_deg=a0, a_end_deg=a1, n_cycles=int(n), total_time=T):
            print(f.summary())


def cmd_design(args) -> None:
    from .control.analysis import analyse, pd_stability_region

    cfg = _apply_overrides(_cfg(args), args)
    p = plant_from(cfg)
    Ts = cfg["loop"]["Ts"]
    lat = cfg["sim"]["sensor"]["latency"] + cfg["sim"]["driver"]["latency"]
    tau = cfg["sim"]["driver"]["tau"]
    print(f"plant: omega0={p.omega0:.4f} rad/s alpha={p.alpha:.3f} beta={p.beta:.4f} 1/m, unstable pole {p.unstable_pole:.3f} 1/s")
    print(f"loop: Ts={Ts * 1e3:.1f} ms, design latency={cfg['loop']['latency_estimate'] * 1e3:.1f} ms, design lag={cfg['loop']['actuator_tau'] * 1e3:.1f} ms")
    print(f"evaluation plant: latency {lat * 1e3:.1f} ms, driver lag {tau * 1e3:.1f} ms")
    reports = {}
    for name in args.algos.split(","):
        ctl = make_controller(name, p, cfg)
        print(f"\n== {name}: {ctl.label} ==")
        _print_dict(ctl.describe())
        if name == "pd":
            _print_dict(pd_stability_region(p))
        law = ctl.law()
        if law is None:
            print("   (nonlinear controller - no linear analysis)")
            continue
        rep = analyse(p, law, Ts, lat, tau)
        reports[name] = rep
        for ln in rep.lines():
            print("   " + ln)
        if abs(rep.spectral_radius - 1.0) < 1e-6:
            print("   NOTE: poles at z = 1 -> cart position/velocity not controlled (drifts)")
    if args.png and reports:
        from .tools.plotting import plot_poles

        plot_poles(reports, args.png)
        print(f"pole map: {args.png}")


# -------------------------------------------------------------- simulation
def _apply_overrides(cfg: dict, a) -> dict:
    cfg = copy.deepcopy(cfg)
    for key, path in (
        ("Ts", ("loop", "Ts")), ("design_latency", ("loop", "latency_estimate")), ("design_tau", ("loop", "actuator_tau")),
        ("duration", ("sim", "duration")), ("theta0", ("sim", "theta0_deg")), ("latency", ("sim", "driver", "latency")),
        ("sensor_latency", ("sim", "sensor", "latency")), ("tau", ("sim", "driver", "tau")), ("backlash", ("sim", "driver", "backlash")),
        ("jitter", ("sim", "driver", "jitter")), ("noise", ("sim", "sensor", "noise_counts")),
        ("offset_counts", ("sim", "calib_error", "adc_upright_offset")), ("estimator", ("estimator", "type")),
        ("seed", ("sim", "seed")), ("x_source", ("sim", "x_source")),
    ):
        v = getattr(a, key, None)
        if v is not None:
            d = cfg
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = v
    if getattr(a, "no_bias", False):
        cfg["estimator"]["estimate_bias"] = False
    if getattr(a, "coulomb", None) is not None:
        cfg["plant"]["coulomb_gamma"] = a.coulomb
    return cfg


def _run_sim(name: str, cfg: dict, kicks=()):
    from .sim.simulator import Disturbance, Simulation

    if name == "swingup":
        cfg = swingup_overrides(cfg)
    p = plant_from(cfg)
    sc = sim_config_from(cfg)
    if name == "swingup":
        sc.theta0_deg = 179.0 if cfg["sim"]["theta0_deg"] == 5.0 else cfg["sim"]["theta0_deg"]
    sc.disturbances = [Disturbance(t, "omega_kick", v) for t, v in kicks]
    sim = Simulation(p, sc, make_controller(name, p, cfg), make_estimator(p, cfg, name))
    return sim.run()


def _kicks(spec: list[str] | None):
    out = []
    for s in spec or []:
        t, v = s.split(":")
        out.append((float(t), float(v)))
    return out


def cmd_simulate(args) -> None:
    cfg = _apply_overrides(_cfg(args), args)
    r = _run_sim(args.algo, cfg, _kicks(args.kick))
    print(f"== {args.algo} ==")
    _print_dict(r.metrics())
    for e in r.events:
        print("   event: " + e)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        r.to_csv(args.csv)
        print(f"csv: {args.csv}")
    if args.png:
        from .tools.plotting import plot_results

        plot_results({args.algo: r}, args.png, f"{args.algo}, theta0={cfg['sim']['theta0_deg']} deg")
        print(f"figure: {args.png}")


def cmd_compare(args) -> None:
    cfg = _apply_overrides(_cfg(args), args)
    results = {}
    print(f"{'algo':8s} {'ok':>5s} {'settle[s]':>9s} {'max|th|':>8s} {'max|x|cm':>9s} {'rms th':>7s} {'rms a':>6s}  trip")
    for name in args.algos.split(","):
        r = _run_sim(name, cfg, _kicks(args.kick))
        results[name] = r
        m = r.metrics()
        print(f"{name:8s} {str(m['balanced']):>5s} {m['settling_time_theta_s']:9.2f} {m['max_abs_theta_deg']:8.2f} "
              f"{m['max_abs_x_cm']:9.2f} {m['rms_theta_last2s_deg']:7.3f} {m['rms_accel']:6.2f}  {m['trip_reason']}")
        if args.csv_dir:
            Path(args.csv_dir).mkdir(parents=True, exist_ok=True)
            r.to_csv(str(Path(args.csv_dir) / f"sim_{name}.csv"))
    if args.png:
        from .tools.plotting import plot_results

        plot_results(results, args.png, "algorithm comparison (same digital twin, same disturbance)")
        print(f"figure: {args.png}")


def cmd_gui(args) -> None:
    from .gui.app import main as gui_main

    gui_main(getattr(args, "config", None))


# ---------------------------------------------------------------- hardware
def _calib_or_default(cfg):
    from .hw.sensor import AngleCalibration

    c = cfg["hardware"]["calibration"]
    return AngleCalibration(c.get("adc_upright") or 1966.0, c.get("counts_per_rad") or 676.7, c.get("sign") or 1)


def _hw_overrides(cfg, args):
    hw = cfg["hardware"]
    for k, attr in (("motor_protocol", "protocol"), ("motor_baud", "motor_baud"), ("sensor_baud", "sensor_baud")):
        v = getattr(args, attr, None)
        if v:
            hw[k] = v
    return hw


def cmd_diagnose(args) -> None:
    from .tools.hwtools import diagnose_motor, diagnose_sensor

    cfg = _cfg(args)
    hw = _hw_overrides(cfg, args)
    if not args.sensor and not args.motor:
        sys.exit("give --sensor COMx and/or --motor COMy")
    if args.sensor:
        diagnose_sensor(args.sensor, hw["sensor_baud"], args.seconds, _calib_or_default(cfg))
    if args.motor:
        diagnose_motor(args.motor, hw["motor_baud"], hw["motor_address"], hw["motor_protocol"], args.seconds)


def cmd_probe(args) -> None:
    from .tools.hwtools import probe_motor

    cfg = _cfg(args)
    hw = _hw_overrides(cfg, args)
    probe_motor(args.motor, hw["motor_baud"], hw["motor_address"], hw["motor_protocol"], args.rate_hz, args.count, args.csv)


def cmd_sensor_log(args) -> None:
    from .hw.sensor import sensor_statistics
    from .tools.hwtools import open_serial, record_sensor, save_sensor_csv

    cfg = _cfg(args)
    hw = _hw_overrides(cfg, args)
    port = open_serial(args.sensor, hw["sensor_baud"])
    print(f"recording {args.seconds:.0f} s from {args.sensor} ...")
    try:
        data = record_sensor(port, args.seconds)
    finally:
        port.close()
    save_sensor_csv(data, args.csv)
    _print_dict(sensor_statistics(data))
    print(f"csv: {args.csv}")


def cmd_calibrate_angle(args) -> None:
    from .tools.hwtools import calibrate_angle_interactive, update_config_file

    cfg = _cfg(args)
    hw = _hw_overrides(cfg, args)
    res = calibrate_angle_interactive(args.sensor, hw["sensor_baud"])
    _print_dict(res)
    if args.write:
        update_config_file(args.write, ["hardware", "calibration"], res)
        print(f"written to {args.write} -> hardware.calibration")


def _driver(cfg, args):
    from .hw.motor import open_driver

    hw = _hw_overrides(cfg, args)
    return open_driver(hw, args.motor)


def cmd_calibrate_cart(args) -> None:
    from .tools.hwtools import cart_calibrate_interactive, update_config_file

    cfg = _cfg(args)
    drv, port = _driver(cfg, args)
    try:
        ok, problems, _ = drv.readiness()
        if not ok:
            sys.exit("driver not ready: " + "; ".join(problems) + "  (use: pendulum-lab driver prepare)")
        res = cart_calibrate_interactive(drv, args.rpm, args.seconds)
    finally:
        port.close()
    _print_dict(res)
    if args.write:
        update_config_file(args.write, ["hardware", "cart"], res)
        print(f"written to {args.write} -> hardware.cart")


def cmd_motor_step(args) -> None:
    from .hw.motor import CartUnits
    from .tools.hwtools import motor_step_test

    cfg = _cfg(args)
    units = CartUnits.from_config(cfg["hardware"])
    drv, port = _driver(cfg, args)
    try:
        ok, problems, _ = drv.readiness()
        if not ok:
            sys.exit("driver not ready: " + "; ".join(problems))
        for sgn in (1, -1):  # out and back
            motor_step_test(drv, units, sgn * args.v, args.seconds, cfg["hardware"]["motor_accel"])
    finally:
        port.close()


def cmd_driver(args) -> None:
    """Official-software actions from the command line (one action per call)."""
    cfg = _cfg(args)
    drv, port = _driver(cfg, args)
    try:
        a = args.action
        if a == "info":
            _print_dict({"system": drv.read_system(), "drive": drv.read_drive(), "homing": drv.read_homing()})
        elif a == "enable":
            drv.enable()
        elif a == "disable":
            drv.disable()
        elif a == "brake":
            drv.brake()
        elif a == "clear":
            drv.clear_state()
        elif a == "recover":
            drv.recover()
        elif a == "speed-mode":
            drv.set_mode(1)
        elif a == "prepare":  # speed mode + echo on + clear + enable + zero speed
            drv.set_mode(1)
            drv.set_echo(True)
            drv.recover()
            drv.set_speed(0.0, 0)
        elif a == "zero":
            drv.zero_position()
        elif a == "limits-on":
            drv.set_limit_switches(True)
        elif a == "limits-off":
            drv.set_limit_switches(False)
        elif a == "save":
            drv.save_params()
        ok, problems, _ = drv.readiness()
        print(f"{a}: done. readiness: " + ("OK" if ok else "; ".join(problems)))
    finally:
        port.close()


def cmd_run(args) -> None:
    import time

    from .hw.motor import make_worker, open_driver
    from .hw.runtime import HardwareLoop, RunOptions, calibration_from
    from .hw.sensor import SensorReader
    from .tools.hwtools import open_serial

    cfg = _cfg(args)
    hw = cfg["hardware"]
    if args.Ts:
        cfg["loop"]["Ts"] = args.Ts
    calib = calibration_from(hw)
    motor = None
    mport = None
    if args.mode == "closed":
        drv, mport = open_driver(hw, args.motor)
        motor = make_worker(drv, hw)
        motor.start()
        ans = input("CLOSED LOOP: cart WILL move. 12 V switch within reach, rail clear, rod held? type YES: ")
        if ans.strip() != "YES":
            motor.stop()
            mport.close()
            sys.exit("aborted")
    sport = open_serial(args.sensor or hw["sensor_port"], hw["sensor_baud"])
    sensor = SensorReader(sport)
    sensor.start()
    log = args.log or str(Path(hw["log_dir"]) / f"run_{args.algo}_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    loop = HardwareLoop(cfg, sensor, motor, RunOptions(args.algo, args.mode, args.duration, log_path=log,
                                                        adapt=args.adapt), calib)
    try:
        msg = loop.run()
    finally:
        if motor:
            motor.stop()
        sensor.stop()
        sport.close()
        if mport:
            mport.close()
    print(f"finished: {msg}; timer overruns: {getattr(loop, 'overruns', 0)}")


def cmd_sil(args) -> None:
    """Hardware code path against the fake rig (no hardware needed)."""
    import threading
    import time

    from .hw.fake import FakeRig
    from .hw.motor import fake_units, make_worker
    from .hw.pd42s1 import PD42S1
    from .hw.runtime import HardwareLoop, RunOptions
    from .hw.sensor import AngleCalibration, SensorReader

    cfg = _cfg(args)
    hw = cfg["hardware"]
    proto = args.protocol or hw["motor_protocol"]
    p = plant_from(cfg)
    if args.true_omega0:
        from dataclasses import replace

        rig_p = replace(p, omega0=args.true_omega0)
    else:
        rig_p = p
    rig = FakeRig(rig_p, proto, theta0=math.radians(1.0))
    rig.start()
    sensor = SensorReader(rig.sensor_port)
    sensor.start()
    motor = make_worker(PD42S1(rig.motor_port, proto, timeout=0.02), hw, fake_units())
    motor.start()
    loop = HardwareLoop(cfg, sensor, motor, RunOptions(args.algo, "closed", args.duration, log_path=args.log,
                                                        adapt=args.adapt), AngleCalibration())
    th = threading.Thread(target=loop.run)
    th.start()
    while loop.live.state not in ("ARMED", "ACTIVE", "DONE"):
        time.sleep(0.01)
    time.sleep(0.3)  # the simulated hand lets go a little after the "release now" prompt
    rig.release()
    th.join()
    print(f"fake rig ({proto}): theta={math.degrees(rig.theta):+.2f} deg, x={rig.x * 100:+.2f} cm, crashed={rig.crashed}, "
          f"motor errors={motor.total_errors}, overruns={loop.overruns}")
    if loop.adapt_log:
        print(f"online identification: omega0_hat = {loop.adapt_log[-1][1]:.4f} rad/s (rig true {rig_p.omega0:.4f})")
    motor.stop()
    sensor.stop()
    rig.stop()


def cmd_report(args) -> None:
    from .tools.plotting import plot_hw_log

    png = args.png or str(Path(args.log).with_suffix(".png"))
    _print_dict(plot_hw_log(args.log, png))
    print(f"figure: {png}")


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pendulum-lab", description="Inverted pendulum computer-control teaching platform")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", help="JSON config merged over defaults (e.g. config/physical_config.json)")
        p.set_defaults(fn=fn)
        return p

    def sim_opts(p):
        p.add_argument("--Ts", type=float, help="sampling period [s]")
        p.add_argument("--duration", type=float)
        p.add_argument("--theta0", type=float, help="initial angle [deg]")
        p.add_argument("--latency", type=float, help="actuation latency in the twin [s]")
        p.add_argument("--sensor-latency", dest="sensor_latency", type=float, help="sensor latency in the twin [s]")
        p.add_argument("--tau", type=float, help="driver velocity-loop lag in the twin [s]")
        p.add_argument("--backlash", type=float, help="belt play [m]")
        p.add_argument("--jitter", type=float, help="command latency jitter std [s]")
        p.add_argument("--noise", type=float, help="ADC noise std [counts]")
        p.add_argument("--offset-counts", dest="offset_counts", type=float, help="upright-zero calibration error [counts]")
        p.add_argument("--coulomb", type=float, help="pivot Coulomb friction gamma [rad/s^2]")
        p.add_argument("--estimator", choices=["kalman", "derivative"])
        p.add_argument("--no-bias", dest="no_bias", action="store_true", help="Kalman filter without bias state")
        p.add_argument("--design-latency", dest="design_latency", type=float, help="latency assumed by LQR/KF design [s]")
        p.add_argument("--design-tau", dest="design_tau", type=float, help="lag assumed by LQR design [s]")
        p.add_argument("--x-source", dest="x_source", choices=["encoder", "command"])
        p.add_argument("--seed", type=int)
        p.add_argument("--kick", action="append", help="rod rate kick 't:rad_per_s', repeatable")

    add("params", cmd_params, "physical parameters derived from the measurements")

    p = add("identify", cmd_identify, "fit omega0/damping from a free swing")
    p.add_argument("--summary", nargs=4, type=float, metavar=("A0_DEG", "AEND_DEG", "CYCLES", "TIME_S"), default=[90, 50, 20, 24.85])
    p.add_argument("--csv", help="sensor-log CSV of a free swing")
    p.add_argument("--min-amp", dest="min_amp", type=float, default=5.0)
    p.add_argument("--write", help="write fitted plant parameters into this config JSON")

    p = add("design", cmd_design, "controller gains, closed-loop poles, delay margin")
    p.add_argument("--algos", default="pd,pid,lqr,place")
    p.add_argument("--png", help="save z-plane pole map")
    sim_opts(p)

    p = add("simulate", cmd_simulate, "run one algorithm in the digital twin")
    p.add_argument("--algo", default="lqr", choices=ALGORITHMS)
    p.add_argument("--csv")
    p.add_argument("--png")
    sim_opts(p)

    p = add("compare", cmd_compare, "run several algorithms on the same scenario")
    p.add_argument("--algos", default="pd,pid,lqr,place")
    p.add_argument("--png")
    p.add_argument("--csv-dir", dest="csv_dir")
    sim_opts(p)

    add("gui", cmd_gui, "interactive digital twin / hardware GUI")

    def hw_opts(p, motor_required=False):
        p.add_argument("--motor", required=motor_required, help="driver COM port, e.g. COM11")
        p.add_argument("--protocol", choices=["custom", "modbus"], help="PD42S1 protocol (as set in the official software)")
        p.add_argument("--motor-baud", dest="motor_baud", type=int)
        p.add_argument("--sensor-baud", dest="sensor_baud", type=int)

    p = add("diagnose", cmd_diagnose, "read-only check of sensor and/or driver")
    p.add_argument("--sensor")
    p.add_argument("--seconds", type=float, default=5.0)
    hw_opts(p)

    p = add("probe", cmd_probe, "read-only communication error-rate test of the PD42S1")
    p.add_argument("--rate-hz", dest="rate_hz", type=float, default=5.0)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--csv")
    hw_opts(p, True)

    p = add("sensor-log", cmd_sensor_log, "record raw ADC stream to CSV (free swing, noise)")
    p.add_argument("--sensor", required=True)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--csv", required=True)
    hw_opts(p)

    p = add("calibrate-angle", cmd_calibrate_angle, "guided angle calibration (hanging/upright/sign)")
    p.add_argument("--sensor", required=True)
    p.add_argument("--write", help="config JSON to update")
    hw_opts(p)

    p = add("calibrate-cart", cmd_calibrate_cart, "define +x and measure belt travel per motor turn (MOVES the cart a few cm)")
    p.add_argument("--rpm", type=float, default=20.0)
    p.add_argument("--seconds", type=float, default=1.5)
    p.add_argument("--write")
    hw_opts(p, True)

    p = add("motor-step", cmd_motor_step, "measure command latency and velocity-loop lag (MOVES the cart ~2 cm)")
    p.add_argument("--v", type=float, default=0.05)
    p.add_argument("--seconds", type=float, default=0.4)
    hw_opts(p, True)

    p = add("driver", cmd_driver, "driver actions like the official software")
    p.add_argument("action", choices=["info", "prepare", "enable", "disable", "brake", "clear", "recover", "speed-mode",
                                      "zero", "limits-on", "limits-off", "save"])
    hw_opts(p, True)

    p = add("run", cmd_run, "real-time control on the rig (shadow or closed)")
    p.add_argument("--algo", default="lqr", choices=[a for a in ALGORITHMS if a != "swingup"])
    p.add_argument("--mode", default="shadow", choices=["shadow", "closed"])
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--sensor")
    p.add_argument("--motor")
    p.add_argument("--Ts", type=float)
    p.add_argument("--log")
    p.add_argument("--adapt", choices=["off", "monitor", "update"], default="monitor",
                   help="online identification of omega0: off / monitor only / update controller gains")

    p = add("sil", cmd_sil, "hardware code path against a simulated rig (no hardware)")
    p.add_argument("--algo", default="lqr", choices=["pd", "pid", "lqr", "place"])
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--protocol", choices=["custom", "modbus"])
    p.add_argument("--true-omega0", dest="true_omega0", type=float, help="rig omega0 different from the design model")
    p.add_argument("--adapt", choices=["off", "monitor", "update"], default="monitor")
    p.add_argument("--log")

    p = add("report", cmd_report, "plot and summarise a hardware run log")
    p.add_argument("--log", required=True)
    p.add_argument("--png")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
