"""Lab configuration: one JSON file, deep-merged over built-in defaults.

``null`` in the hardware section means "must be measured on the rig"; the
hardware runtime refuses to move the motor while any required value is null.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .control.actuator import ActuatorLimits
from .control.controllers import LQR, AnglePD, CascadePID, Controller, EnergySwingUp, NullController, PolePlacement
from .control.estimators import DirtyDerivative, Estimator, HybridEstimator, KalmanEstimator, SwingEstimator
from .control.safety import SafetyLimits
from .model.params import PlantParams, default_params

ROOT = Path(__file__).resolve().parent.parent

ALGORITHMS = ("pd", "pid", "lqr", "place", "swingup", "none")


# Free-swing recording on the rig, 2026-09-27 (tests/data/swing_20260927_151631.csv):
# release from 90.5 deg, 60 s, 97 full cycles; method A (turning points + energy
# balance, sensor self-calibrated). Method B (SVF + LS) gives omega0 within 0.1 %.
MEASURED_PLANT = {"omega0": 5.5147, "viscous_c": 0.0363, "coulomb_gamma": 0.0, "quad_d": 0.00297}


def _plant_default() -> dict:
    p = default_params()
    return {
        **MEASURED_PLANT,
        "mass": round(p.mass, 4),
        "l_c": round(p.l_c, 4),
        "J": round(p.J, 6),
        "cart_mass": 0.12,
        "g": 9.81,
        "source": "free swing 2026-09-27, method A (geometry for mass/l_c/J)",
    }


DEFAULTS: dict = {
    "plant": _plant_default(),
    "loop": {
        "Ts": 0.005,
        "latency_estimate": 0.020,   # s, total sensor->motion latency assumed in design
        "actuator_tau": 0.015,       # s, driver velocity-loop lag assumed in design
    },
    "actuator": {"a_max": 6.0, "v_max": 0.5, "v_quantum": 0.0},
    "safety": {
        "theta_trip_deg": 25.0,
        "x_soft": 0.20,
        "a_brake": 3.0,
        "latency": 0.03,
        "v_max": 0.5,
        "sensor_timeout": 0.05,
        "max_motor_errors": 3,
        "max_runtime": 300.0,
        "adc_min_valid": 20,
        "adc_max_valid": 4075,
        "check_adc_range": True,
    },
    "estimator": {
        "type": "kalman",            # 'kalman' | 'derivative'
        "estimate_bias": True,
        "sigma_theta": 1.5e-3,
        "sigma_x": 2e-4,
        "sigma_a": 0.5,
        "sigma_alpha": 3.0,
        "sigma_bias_rw": 2e-4,
        "sigma_bias0_deg": 3.0,
        "derivative_cutoff_hz": 20.0,
    },
    "controllers": {
        "pd": {"wn": 8.0, "zeta": 1.0, "kp": None, "kd": None},
        "pid": {"wn": 8.0, "zeta": 1.0, "wo": 0.8, "zeta_o": 1.0, "ki": 0.0, "theta_ref_max_deg": 6.0},
        "lqr": {"q": [20.0, 5.0, 200.0, 5.0], "r": 0.2, "delay_aware": True, "include_lag": True, "q_integral": 0.0},
        "place": {"poles": [[-7.0, 5.0], [-7.0, -5.0], [-3.0, 0.0], [-2.0, 0.0]], "delay_aware": True, "include_lag": True},
        "swingup": {"k_energy": 0.06, "a_max": 3.0, "kx": 25.0, "kv": 6.0, "catch_deg": 25.0},
    },
    "sim": {
        "duration": 10.0,
        "theta0_deg": 5.0,
        "substeps": 10,
        "rail_half": 0.30,
        "x_source": "encoder",
        "driver": {"tau": 0.015, "accel_limit": 20.0, "latency": 0.012, "jitter": 0.002, "backlash": 0.0},
        "sensor": {"adc_upright": 1966.0, "counts_per_rad": 678.1, "sign": 1, "noise_counts": 0.6,
                   "latency": 0.006, "x_resolution": 2.5e-6, "x_latency": 0.010},
        "calib_error": {"adc_upright_offset": 0.0, "counts_per_rad_scale": 1.0},
        "seed": 1,
    },
    "hardware": {
        "sensor_port": "COM12",
        "sensor_baud": 115200,
        "motor_port": "COM11",
        "motor_baud": 115200,
        "motor_address": 1,
        "motor_protocol": "custom",          # 'custom' (C5..5C, shorter frames) or 'modbus' (RTU)
        "motor_accel": 0,                    # F1 accel byte: 0 = follow each command immediately
        "calibration": {"adc_upright": None, "counts_per_rad": None, "sign": None},
        "cart": {
            "meters_per_rev": None,          # belt travel per motor turn (calibrate-cart)
            "center_counts": None,           # encoder at rail centre
            "sign": 1,                       # encoder direction for 正转
        },
        "motor_feedback_every": 2,           # read position every N control periods
        "log_dir": "data",
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = deep_merge(cfg, user)
    return cfg


def plant_from(cfg: dict) -> PlantParams:
    return PlantParams.from_dict(cfg["plant"])


def safety_from(cfg: dict) -> SafetyLimits:
    s = dict(cfg["safety"])
    th = s.pop("theta_trip_deg")
    return SafetyLimits(theta_trip=math.radians(th), **s)


def actuator_from(cfg: dict) -> ActuatorLimits:
    return ActuatorLimits(**cfg["actuator"])


def make_estimator(p: PlantParams, cfg: dict, algorithm: str | None = None) -> Estimator:
    e, loop = cfg["estimator"], cfg["loop"]
    kf = KalmanEstimator(
        p, loop["Ts"], delay=loop["latency_estimate"],
        sigma_theta=e["sigma_theta"], sigma_x=e["sigma_x"], sigma_a=e["sigma_a"], sigma_alpha=e["sigma_alpha"],
        estimate_bias=e["estimate_bias"], sigma_bias_rw=e["sigma_bias_rw"],
        sigma_bias0=math.radians(e["sigma_bias0_deg"]),
    )
    if algorithm == "swingup":
        return HybridEstimator(SwingEstimator(loop["Ts"]), kf)
    if e["type"] == "derivative":
        return DirtyDerivative(loop["Ts"], cutoff_hz=e["derivative_cutoff_hz"])
    return kf


def swingup_overrides(cfg: dict) -> dict:
    """Swing-up passes through the hanging position: no angle trip, ADC ends are expected."""
    c = copy.deepcopy(cfg)
    c["safety"]["theta_trip_deg"] = 1e6
    c["safety"]["check_adc_range"] = False
    c["safety"]["x_soft"] = max(c["safety"]["x_soft"], 0.25)
    return c


def make_controller(name: str, p: PlantParams, cfg: dict) -> Controller:
    loop, cc = cfg["loop"], cfg["controllers"]
    Ts = loop["Ts"]
    if name == "pd":
        c = cc["pd"]
        if c.get("kp") is not None and c.get("kd") is not None:
            return AnglePD(c["kp"], c["kd"])
        return AnglePD.from_poles(p, c["wn"], c["zeta"])
    if name == "pid":
        c = cc["pid"]
        ctl = CascadePID.design(p, Ts, c["wn"], c["zeta"], c["wo"], c["zeta_o"], c["ki"])
        ctl.theta_ref_max = math.radians(c["theta_ref_max_deg"])
        return ctl
    if name == "lqr":
        c = cc["lqr"]
        return LQR(
            p, Ts, q=c["q"], r=c["r"],
            design_delay=loop["latency_estimate"] if c["delay_aware"] else 0.0,
            actuator_tau=loop["actuator_tau"] if c["include_lag"] else 0.0,
            q_integral=c["q_integral"],
        )
    if name == "place":
        c = cc["place"]
        poles = [complex(re, im) for re, im in c["poles"]]
        return PolePlacement(
            p, Ts, poles,
            design_delay=loop["latency_estimate"] if c["delay_aware"] else 0.0,
            actuator_tau=loop["actuator_tau"] if c["include_lag"] else 0.0,
        )
    if name == "swingup":
        c = cc["swingup"]
        return EnergySwingUp(p, make_controller("lqr", p, cfg), k_energy=c["k_energy"], a_max=c["a_max"],
                             kx=c["kx"], kv=c["kv"], catch_angle=math.radians(c["catch_deg"]))
    if name == "none":
        return NullController()
    raise ValueError(f"unknown algorithm '{name}', choose from {ALGORITHMS}")


def sim_config_from(cfg: dict, **over):
    from .sim.simulator import Calibration, DriverModel, SensorModel, SimConfig

    s = cfg["sim"]
    sensor = SensorModel(**s["sensor"])
    ce = s["calib_error"]
    calib = Calibration(
        adc_upright=sensor.adc_upright + ce["adc_upright_offset"],
        counts_per_rad=sensor.counts_per_rad * ce["counts_per_rad_scale"],
        sign=sensor.sign,
    )
    sc = SimConfig(
        Ts=cfg["loop"]["Ts"],
        substeps=s["substeps"],
        duration=s["duration"],
        theta0_deg=s["theta0_deg"],
        rail_half=s["rail_half"],
        x_source=s["x_source"],
        driver=DriverModel(**s["driver"]),
        sensor=sensor,
        calib=calib,
        act=actuator_from(cfg),
        safety=safety_from(cfg),
        seed=s["seed"],
    )
    for k, v in over.items():
        setattr(sc, k, v)
    return sc
