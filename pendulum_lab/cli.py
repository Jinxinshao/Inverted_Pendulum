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
    """Free-swing CSV -> sensor self-calibration + two independent identification methods.

    A  turning points (one per valid segment) + half-cycle energy balance, with the
       potentiometer mapping calibrated from the swing itself (recommended);
    B  state-variable filter (FOH, corrected) + least squares on the ODE over all
       valid samples.
    Returns {"analysis"/"A": SwingAnalysis, "B": LSFit, "rest": (median, std, outliers)}.
    """
    import numpy as np

    from .control.online_id import ls_free_swing
    from .model.identification import fit_decay_summary_both
    from .model.swing_analysis import analyse_swing, robust_rest
    from .tools.hwtools import load_sensor_csv, split_rest_swing

    cal = _calib_or_default(cfg)
    t, adc, phase = load_sensor_csv(path, with_phase=True)
    rest, swing = split_rest_swing(t, phase)
    fits: dict = {}
    rest_adc = None
    if rest.sum() > 20:
        rest_adc, rest_sd, n_out = robust_rest(adc[rest])
        fits["rest"] = (rest_adc, rest_sd, n_out)
        out(f"静止下垂: 读数中位数 {rest_adc:.1f}, 噪声(MAD) {rest_sd:.2f} LSB, 跳变点 {n_out} 个"
            + ("（下垂位置落在电位器端点附近，偶发读数跳到另一端，属正常）" if n_out else ""))
    r = analyse_swing(t[swing], adc[swing], rest_adc, min_amp_deg=min_amp)
    fits["analysis"] = fits["A"] = r
    for ln in r.lines():
        out(ln)
    c = r.calibration.describe()
    d_up = c["implied_upright_adc"] - cal.adc_upright
    out(f"  配置中的直立读数 {cal.adc_upright:.1f}：相差 {d_up:+.1f} LSB = {np.degrees(d_up / r.calibration.K):+.2f}°"
        "（电位器线性度 ±0.5 % 时外推 180° 的不确定度约 ±10 LSB；最终以卡尔曼零偏估计为准）")
    try:
        fb = ls_free_swing(r.t, r.phi, wf=40.0, t_start=float(r.t_ext[0]), drag=True)
        fits["B"] = fb
    except ValueError as e:
        fb = None
        out(f"方法 B 失败: {e}")
    out("")
    out("== 方法比较 ==")
    out(f"方法 A（转折点 + 能量平衡，推荐）  ω0 = {r.omega0:.4f} rad/s  c={r.viscous_c:.4f} γ={r.coulomb_gamma:.4f} d={r.quad_d:.5f}")
    if fb is not None:
        dev = (fb.omega0 - r.omega0) / r.omega0 * 100
        out(f"方法 B（状态变量滤波 + 最小二乘）  ω0 = {fb.omega0:.4f} rad/s  c={fb.viscous_c:.4f} γ={fb.coulomb_gamma:.4f} "
            f"d={fb.quad_d:.5f}   ω0 偏差 {dev:+.2f} %")
        out("  B 的 ω0 可信；B 的阻尼项不可信：死区正好切掉了每次摆过最低点时速度最大的一段，"
            "粘性/库仑/空气阻力三个回归量在剩下的数据里高度相关。A 用能量平衡积分了整个半周期，不受影响。")
    v, cc = fit_decay_summary_both()
    out(f"秒表法（20 次 24.85 s，90°→50°）: 粘性模型 {v.omega0:.4f} / 库仑模型 {cc.omega0:.4f} rad/s")
    return fits


def cmd_identify(args) -> None:
    from .model.identification import fit_decay_summary_both
    from .tools.hwtools import update_config_file

    if args.csv:
        cfg = _cfg(args)
        fits = identify_csv(args.csv, cfg, args.min_amp)
        fe = fits["A"]
        if args.write:
            update_config_file(args.write, ["plant"], {
                "omega0": round(fe.omega0, 5), "viscous_c": round(fe.viscous_c, 5),
                "coulomb_gamma": round(fe.coulomb_gamma, 5), "quad_d": round(fe.quad_d, 6),
                "source": f"free swing {args.csv} (method A)"})
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
    return AngleCalibration(c.get("adc_upright") or 1966.0, c.get("counts_per_rad") or 678.1, c.get("sign") or 1)


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


def cmd_drivertest(args) -> None:
    """Guided, safe answers to the open driver questions (0 rpm, Modbus padding, limit switches)."""
    from .tools.drivertest import DriverSelfTest

    cfg = _cfg(args)
    steps = tuple(args.steps.split(","))
    if args.sim:
        import math

        from .hw.fake import FakeRig
        from .hw.pd42s1 import PD42S1

        protocol = args.protocol or "modbus"
        rig = FakeRig(plant_from(cfg), protocol, theta0=math.pi, mode=2)  # starts like the real one: torque mode
        rig.start()
        drv = PD42S1(rig.motor_port, protocol, timeout=0.05)
        print("== 演练模式：驱动器模拟器（没有限位开关，第 3 步应显示“没停”） ==")
        try:
            DriverSelfTest(drv, mm_per_rev=args.mm_per_rev, rpm_move=args.rpm, rpm_limit=args.rpm_limit,
                           log_dir=cfg["hardware"]["log_dir"]).run(steps)
        finally:
            rig.stop()
        return
    drv, port = _driver(cfg, args)
    try:
        DriverSelfTest(drv, mm_per_rev=args.mm_per_rev, rpm_move=args.rpm, rpm_limit=args.rpm_limit,
                       log_dir=cfg["hardware"]["log_dir"], config_path=args.write).run(steps)
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
    motor = make_worker(PD42S1(rig.motor_port, proto, timeout=0.05), hw, fake_units())  # same as open_driver
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

    p = add("drivertest", cmd_drivertest, "guided driver self-test: 0 rpm stop, Modbus padding, limit switches (slow, safe)")
    p.add_argument("--sim", action="store_true", help="rehearse against the driver emulator (no hardware)")
    p.add_argument("--steps", default="0,1,2,3,4", help="subset of steps, e.g. 0,1,2")
    p.add_argument("--rpm", type=float, default=20.0, help="speed for the 0-rpm stop test")
    p.add_argument("--rpm-limit", dest="rpm_limit", type=float, default=10.0, help="speed for the limit-switch test")
    p.add_argument("--mm-per-rev", dest="mm_per_rev", type=float, default=40.0, help="belt travel estimate until calibrated")
    p.add_argument("--write", help="config JSON to store the belt calibration (step 4)")
    hw_opts(p)

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
