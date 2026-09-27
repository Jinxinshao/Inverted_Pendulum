"""Quantitative comparison of control algorithms and identification methods.

    python examples/benchmark.py            -> docs/benchmark_results.md (+ printed tables)

Everything runs in the digital twin with the same seeds, so the tables are
reproducible and can be regenerated after re-identifying the real rig.
"""
import copy
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pendulum_lab.config import load_config, make_controller, make_estimator, plant_from, sim_config_from  # noqa: E402
from pendulum_lab.control.analysis import delay_margin  # noqa: E402
from pendulum_lab.control.online_id import AdaptiveUpdater, ls_free_swing  # noqa: E402
from pendulum_lab.model.identification import (  # noqa: E402
    extrema, fit_decay_summary, fit_free_decay, fit_free_decay_one_sided, simulate_free_swing,
)
from pendulum_lab.sim.simulator import Disturbance, SensorModel, Simulation  # noqa: E402

CFG = load_config()
P = plant_from(CFG)

ALGOS = [
    ("pd", "PD (angle only)", {}),
    ("pid", "cascade PID", {}),
    ("lqr", "LQR (delay+lag aware)", {}),
    ("lqr", "LQR (naive: no delay/lag)", {"naive": True}),
    ("place", "pole placement", {}),
]

SCENARIOS = [
    ("nominal", "Ts 5 ms, latency 18 ms, lag 15 ms, 5 deg start, kick at 5 s", {}),
    ("latency 35 ms", "actuation latency 12 -> 29 ms (total 35 ms)", {"sim.driver.latency": 0.029}),
    ("Ts 10 ms", "sampling period doubled", {"loop.Ts": 0.010}),
    ("zero error 1 deg", "upright calibration off by 12 LSB, bias state ON", {"sim.calib_error.adc_upright_offset": 12}),
    ("zero error, no bias state", "same, bias state OFF", {"sim.calib_error.adc_upright_offset": 12, "estimator.estimate_bias": False}),
    ("backlash 2 mm", "belt play 2 mm", {"sim.driver.backlash": 0.002}),
    ("model -10 %", "true omega0 = 0.9 x design", {"true_w": 0.9}),
    ("model +10 %", "true omega0 = 1.1 x design", {"true_w": 1.1}),
    ("noise x3", "ADC noise 1.8 LSB", {"sim.sensor.noise_counts": 1.8}),
    ("Coulomb 0.2", "pivot Coulomb friction gamma = 0.2 rad/s^2", {"plant.coulomb_gamma": 0.2}),
]


def _set(cfg, path, v):
    d = cfg
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = v


def run_case(algo, opts, over, duration=12.0):
    cfg = copy.deepcopy(CFG)
    cfg["sim"]["duration"] = duration
    true_w = None
    for k, v in over.items():
        if k == "true_w":
            true_w = v
        else:
            _set(cfg, k, v)
    if opts.get("naive"):
        cfg["controllers"]["lqr"].update(delay_aware=False, include_lag=False)
    p_design = plant_from(cfg)
    p_true = replace(p_design, omega0=p_design.omega0 * true_w) if true_w else p_design
    sc = sim_config_from(cfg)
    sc.disturbances = [Disturbance(5.0, "omega_kick", 0.5)]
    ctl = make_controller(algo, p_design, cfg)
    r = Simulation(p_true, sc, ctl, make_estimator(p_design, cfg)).run()
    m = r.metrics()
    return m


def control_table():
    lines = ["| scenario | " + " | ".join(lbl for _, lbl, _ in ALGOS) + " |",
             "|---|" + "---|" * len(ALGOS)]
    for sname, _desc, over in SCENARIOS:
        cells = []
        for algo, _lbl, opts in ALGOS:
            m = run_case(algo, opts, over)
            if not m["balanced"]:
                why = "rail" if ("soft limit" in m["trip_reason"] or "predicted" in m["trip_reason"]) else "fell"
                cells.append(f"FAIL ({why})")
            else:
                cells.append(f"{m['rms_theta_last2s_deg']:.2f}° / {m['max_abs_x_cm']:.0f} cm / {m['rms_accel']:.2f}")
        lines.append(f"| {sname} | " + " | ".join(cells) + " |")
        print(lines[-1])
    return lines


def margin_table():
    lines = ["| algorithm | delay margin (Ts 5 ms, lag 15 ms) |", "|---|---|"]
    for algo, lbl, opts in ALGOS:
        cfg = copy.deepcopy(CFG)
        if opts.get("naive"):
            cfg["controllers"]["lqr"].update(delay_aware=False, include_lag=False)
        law = make_controller(algo, P, cfg).law()
        dm = delay_margin(P, law, CFG["loop"]["Ts"], 0.015)
        lines.append(f"| {lbl} | {'cart drifts (poles at z=1)' if dm is None else f'{dm * 1e3:.0f} ms'} |")
    return lines


def rig_like_swing(seed=0, c=0.02, gamma=0.1, w=5.55):
    """Twin of the recorder output: 200 Hz, 12-bit, dead zone at the bottom with glitches."""
    t, phi = simulate_free_swing(w, c=c, gamma=gamma, a0=math.radians(88), duration=60, fs=200)
    sm, rng = SensorModel(glitch_prob=0.3), np.random.default_rng(seed)
    adc = np.array([sm.adc_of(th + math.pi, rng) for th in phi], float)
    return t, adc


def _row(name, f, note):
    return f"| {name} | {f.omega0:.4f} | {f.viscous_c:.4f} | {f.coulomb_gamma:.4f} | {getattr(f, 'quad_d', 0.0):.5f} | {note} |"


def _legacy_rows(t, adc):
    """The first version (before the real recording): extrema on the raw angle."""
    from pendulum_lab.hw.sensor import AngleCalibration
    from pendulum_lab.tools.hwtools import swing_angle_from_adc

    phi = swing_angle_from_adc(adc, AngleCalibration())
    out = []
    for name, fn in (("old A1: raw extrema, one side", lambda: fit_free_decay_one_sided(t, phi, -1)),
                     ("old A2: raw extrema, both sides", lambda: fit_free_decay(t, phi))):
        try:
            out.append(_row(name, fn(), "glitches while crossing the dead zone become false extrema"))
        except ValueError as e:
            out.append(f"| {name} | failed | | | | {e} |")
    return out


def ident_table():
    from pendulum_lab.model.swing_analysis import analyse_swing
    from pendulum_lab.tools.hwtools import load_sensor_csv, split_rest_swing

    truth = (5.55, 0.02, 0.10)
    head = ["| method | omega0 [rad/s] | c [1/s] | gamma [rad/s^2] | d [1/rad] | notes |", "|---|---|---|---|---|---|"]
    lines = ["### Digital twin of the recording (truth known)", ""] + head + [
        f"| **truth** | {truth[0]:.4f} | {truth[1]:.4f} | {truth[2]:.4f} | 0 | 88 deg release, 200 Hz, 12-bit, dead zone at the bottom with glitches |"]
    # stopwatch: count 20 cycles on the true signal
    t0, ph0 = simulate_free_swing(truth[0], c=truth[1], gamma=truth[2], a0=math.pi / 2, duration=30)
    te, ve = extrema(t0, ph0)
    pos = [(a, b) for a, b in zip(te, ve) if b > 0]
    T20, A20 = pos[19][0], math.degrees(pos[19][1])
    lines.append(f"| stopwatch, naive T = t/20 | {2 * math.pi * 20 / T20:.4f} | - | - | - | ignores the amplitude dependence of the period |")
    fv = fit_decay_summary(90, A20, 20, T20, "viscous")
    fc = fit_decay_summary(90, A20, 20, T20, "coulomb")
    lines.append(_row("stopwatch + elliptic correction (viscous)", fv, "two numbers cannot separate c and gamma"))
    lines.append(_row("stopwatch + elliptic correction (Coulomb)", fc, ""))
    t, adc = rig_like_swing()
    lines += _legacy_rows(t, adc)
    a = analyse_swing(t, adc, 4092)
    lines.append(_row("**A: turning points + energy balance**", a,
                      f"recommended; damping model by AIC: {a.damping.name}; sensor K {a.calibration.K:.1f} (truth {SensorModel().counts_per_rad})"))
    fb = ls_free_swing(a.t, a.phi, wf=40.0, t_start=float(a.t_ext[0]), drag=True)
    lines.append(_row("B: SVF (FOH, corrected) + least squares", fb, "omega0 good; damping terms collinear (dead zone removes the fastest part)"))
    # the real recording
    t, adc, ph = load_sensor_csv(str(ROOT / "tests" / "data" / "swing_20260927_151631.csv"), with_phase=True)
    rest, sw = split_rest_swing(t, ph)
    a = analyse_swing(t[sw], adc[sw], float(np.median(adc[rest])))
    fb = ls_free_swing(a.t, a.phi, wf=40.0, t_start=float(a.t_ext[0]), drag=True)
    rv = fit_decay_summary(90, 50, 20, 24.85, "viscous")
    lines += ["", "### Real recording 2026-09-27 (tests/data/swing_20260927_151631.csv)", ""] + head
    lines += _legacy_rows(t[sw], adc[sw])
    lines.append(_row("**A: turning points + energy balance**", a,
                      f"{a.damping.name}; + side {a.per_side['+']:.4f}, - side {a.per_side['-']:.4f}; K {a.calibration.K:.1f}"))
    lines.append(_row("B: SVF (FOH, corrected) + least squares", fb, f"omega0 {100 * (fb.omega0 / a.omega0 - 1):+.2f} % vs A"))
    lines.append(_row("stopwatch 90->50 deg, 20 cycles, 24.85 s (viscous)", rv,
                      "the timing was right (recording: 24.82 s); the eye-read end amplitude was 45.8 deg, not 50"))
    lines.append("")
    lines.append("### Closed-loop online identification (twin, true omega0 = 5.55)")
    lines.append("")
    lines += ["| method | omega0 [rad/s] | c | gamma | d | notes |", "|---|---|---|---|---|---|"]
    # online RLS in closed loop
    for label, exc, kicks in (("online RLS, closed loop, known excitation", 0.03, False),
                              ("online RLS, closed loop, no excitation", 0.0, False),
                              ("online RLS, closed loop, hand taps", 0.03, True)):
        cfg = copy.deepcopy(CFG)
        cfg["sim"]["duration"] = 40
        sc = sim_config_from(cfg)
        if kicks:
            sc.disturbances = [Disturbance(tt, "omega_kick", 0.3 * (-1) ** i) for i, tt in enumerate(np.arange(3, 40, 3))]
        ad = AdaptiveUpdater(P, cfg, "lqr", "monitor", excitation_amp=exc)
        Simulation(replace(P, omega0=truth[0]), sc, make_controller("lqr", P, cfg), make_estimator(P, cfg), ad).run()
        note = f"±{ad.rls.omega0_std:.3f} (1σ); rejected {ad.reject_fraction * 100:.1f} %; {'trusted' if ad.trustworthy else 'NOT trusted'}"
        lines.append(f"| {label} | {ad.rls.omega0:.4f} | (fixed) | - | - | {note} |")
    for ln in lines:
        print(ln)
    return lines


if __name__ == "__main__":
    t0 = time.time()
    out = ["# Benchmark results (generated by examples/benchmark.py)", "",
           "## Control algorithms in the digital twin", "",
           "Cell: RMS angle in the last 2 s / max |cart travel| / RMS cart acceleration [m/s²]. "
           "FAIL(rail) = safety stop at the ±20 cm soft limit, FAIL(fell) = rod fell.", ""]
    out += [f"- **{n}**: {d}" for n, d, _ in SCENARIOS]
    out += [""] + control_table() + ["", "### Linear delay margin", ""] + margin_table()
    out += ["", "## Identification methods", ""] + ident_table()
    out += ["", f"_runtime {time.time() - t0:.0f} s_"]
    (ROOT / "docs" / "benchmark_results.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("written docs/benchmark_results.md")
