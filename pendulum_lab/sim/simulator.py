"""Digital twin: nonlinear cart-pendulum + PD42S1/belt + potentiometer + PC loop.

Everything the real loop does is reproduced at the same sampling period:

    plant (RK4, sub-steps) --latency--> ADC quantisation --> calibration --> estimator
      ^                                                                        |
      |                                                                        v
    driver velocity loop <--latency/jitter-- velocity command <-- integrator <-- controller
                                                                  (+ safety supervisor)

Modelled non-idealities (each can be switched off to show its effect):
  sampling period, sensor/actuator latency, command jitter, 12-bit ADC
  quantisation and noise, calibration error of the upright zero, potentiometer
  electrical dead zone (345 deg of 360), driver velocity-loop lag and
  acceleration limit, speed-command quantisation, belt backlash, pivot viscous
  and Coulomb friction, hard rail ends.
"""
from __future__ import annotations

import csv
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..control.actuator import ActuatorLimits, VelocityCommandIntegrator
from ..control.controllers import ControlInput, Controller
from ..control.estimators import Estimator
from ..control.safety import SafetyLimits, Supervisor
from ..model.dynamics import rk4_pendulum, wrap_angle
from ..model.params import PlantParams


@dataclass
class DriverModel:
    """PD42S1 velocity loop + belt as seen from the cart."""

    tau: float = 0.015          # s, velocity-loop time constant
    accel_limit: float = 20.0   # m/s^2 the motor/driver can deliver
    latency: float = 0.008      # s, command transmission + driver reaction
    jitter: float = 0.0         # s, std of additional random command latency
    backlash: float = 0.0       # m, total belt play


@dataclass
class SensorModel:
    """Potentiometer + 12-bit ADC as it really is (the 'truth')."""

    adc_upright: float = 1966.0     # true ADC reading at exactly upright
    counts_per_rad: float = 676.7   # true slope
    sign: int = 1                   # +1: ADC increases when rod leans to +x
    noise_counts: float = 0.6       # std of electrical noise [counts]
    latency: float = 0.004          # s, sampling-to-PC transport
    full_scale: int = 4095
    electrical_deg: float = 345.0   # WDD35D4: 345 deg electrical of 360 mechanical
    x_resolution: float = 2.5e-6    # m, cart position quantum from the motor encoder
    x_latency: float = 0.010        # s, age of the motor position reading

    def adc_of(self, theta: float, rng: np.random.Generator | None) -> int:
        span = self.full_scale / math.radians(self.electrical_deg)  # counts per rad along the track
        phi_up = self.adc_upright / span  # angle along the track at upright [rad]
        # the true slope may differ slightly from the nominal track slope (gain error)
        phi = (phi_up + self.sign * theta * self.counts_per_rad / span) % (2 * math.pi)
        if phi > math.radians(self.electrical_deg):
            adc = float(self.full_scale)  # wiper beyond the track: reads the end terminal
        else:
            adc = phi * span
        if rng is not None and self.noise_counts > 0:
            adc += rng.normal(0.0, self.noise_counts)
        return int(min(self.full_scale, max(0, round(adc))))


@dataclass
class Calibration:
    """What the software BELIEVES about the sensor (see hw/calibration.py)."""

    adc_upright: float = 1966.0
    counts_per_rad: float = 676.7
    sign: int = 1

    def theta_of(self, adc: float) -> float:
        return self.sign * (adc - self.adc_upright) / self.counts_per_rad


@dataclass
class Disturbance:
    t: float
    kind: str = "omega_kick"   # 'omega_kick' [rad/s] | 'theta_step' [rad] | 'x_ref' [m]
    value: float = 0.3


@dataclass
class SimConfig:
    Ts: float = 0.005
    substeps: int = 10
    duration: float = 10.0
    theta0_deg: float = 5.0
    omega0: float = 0.0
    x0: float = 0.0
    x_ref: float = 0.0
    rail_half: float = 0.30           # m, hard end stops (70 cm rail minus cart width)
    x_source: str = "encoder"         # 'encoder' | 'command' (dead reckoning)
    driver: DriverModel = field(default_factory=DriverModel)
    sensor: SensorModel = field(default_factory=SensorModel)
    calib: Calibration = field(default_factory=Calibration)
    act: ActuatorLimits = field(default_factory=ActuatorLimits)
    safety: SafetyLimits = field(default_factory=SafetyLimits)
    disturbances: list[Disturbance] = field(default_factory=list)
    end_after_trip: float = 1.0
    seed: int = 1

    @property
    def total_latency(self) -> float:
        return self.sensor.latency + self.driver.latency


class Simulation:
    """Stepwise simulation (used by the GUI) with :meth:`run` for batch use."""

    LOG_KEYS = (
        "t", "x", "v", "theta", "omega", "adc", "theta_meas", "x_meas",
        "x_hat", "v_hat", "theta_hat", "omega_hat", "bias_hat",
        "a_ctrl", "a_eff", "v_cmd", "v_motor", "x_ref", "tripped", "omega0_hat",
    )

    def __init__(self, p: PlantParams, cfg: SimConfig, controller: Controller, estimator: Estimator, adapter=None):
        """``p`` is the TRUE plant of the twin; the controller may be designed for another model.
        ``adapter``: optional online_id.AdaptiveUpdater (online identification / self-tuning)."""
        self.p, self.cfg, self.ctrl, self.est = p, cfg, controller, estimator
        self.adapter = adapter
        self.ctrl0 = controller
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self) -> None:
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        self.t = 0.0
        self.k = 0
        self.theta = math.radians(c.theta0_deg)
        self.omega = c.omega0
        self.x_c = self.x_m = c.x0
        self.x_dr = c.x0  # dead-reckoned position (integral of sent velocity commands)
        self.v_c = self.v_m = 0.0
        self.x_ref = c.x_ref
        self.v_target = 0.0
        self.cmd_queue: deque = deque()
        self.hist: deque = deque(maxlen=int(0.2 / (c.Ts / c.substeps)) + 4)
        self.hist.append((0.0, self.theta, self.x_m))
        self.integrator = VelocityCommandIntegrator(c.Ts, c.act)
        self.sup = Supervisor(c.safety)
        self.u_hist: deque = deque([0.0] * 32, maxlen=32)
        self.a_prev = 0.0
        self.ctrl = self.ctrl0
        self.ctrl.reset()
        self.est.reset(math.radians(c.theta0_deg), c.x0)
        self.events: list[str] = []
        self.pending = sorted(c.disturbances, key=lambda d: d.t)
        self.log: dict[str, list] = {k: [] for k in self.LOG_KEYS}
        self.finished = False
        self.collided = False

    def kick(self, d_omega: float) -> None:
        """Interactive disturbance (GUI button): instantaneous rod rate change."""
        self.omega += d_omega
        self.events.append(f"t={self.t:.2f}s kick {d_omega:+.2f} rad/s")

    # --------------------------------------------------------------- helpers
    def _delayed(self, latency: float) -> tuple[float, float]:
        target = self.t - latency
        for (tt, th, xm) in reversed(self.hist):
            if tt <= target + 1e-12:
                return th, xm
        return self.hist[0][1], self.hist[0][2]

    def _apply_disturbances(self) -> None:
        while self.pending and self.pending[0].t <= self.t + 1e-12:
            d = self.pending.pop(0)
            if d.kind == "omega_kick":
                self.omega += d.value
            elif d.kind == "theta_step":
                self.theta += d.value
            elif d.kind == "x_ref":
                self.x_ref = d.value
            self.events.append(f"t={self.t:.2f}s {d.kind} {d.value:+.3f}")

    # ------------------------------------------------------------------ step
    def step(self) -> bool:
        """Advance one control period. Returns False when the run is over."""
        if self.finished:
            return False
        c, p = self.cfg, self.p
        self._apply_disturbances()

        # ---- measurement (what the PC receives at t_k)
        th_d, xm_d = self._delayed(c.sensor.latency)
        adc = c.sensor.adc_of(th_d, self.rng)
        theta_meas = c.calib.theta_of(adc)
        if hasattr(self.est, "in_deadzone"):
            self.est.in_deadzone = not (c.safety.adc_min_valid <= adc <= c.safety.adc_max_valid)
        if c.x_source == "encoder":
            _, xm_x = self._delayed(c.sensor.x_latency)
            q = c.sensor.x_resolution
            x_meas = round(xm_x / q) * q if q > 0 else xm_x
        else:
            x_meas = self.x_dr
        est = self.est.update(theta_meas, x_meas, self.integrator.v_cmd, self.a_prev)

        # ---- supervisor + controller
        ok = self.sup.check(self.t, est.x, est.v, est.theta, adc=adc, runtime=self.t)
        if self.adapter is not None and ok:
            new_ctrl = self.adapter.step(self.t, theta_meas - est.bias, self.a_prev, self.ctrl, self.est)
            if new_ctrl is not None:
                self.ctrl = new_ctrl
                self.events.append(f"t={self.t:.2f}s controller redesigned for omega0={self.adapter.p_design.omega0:.4f}")
        x_ref = self.x_ref + (self.adapter.reference(self.t) if self.adapter is not None else 0.0)
        ci = ControlInput(self.t, est.x, est.v, est.theta, est.omega, self.integrator.v_cmd, self.u_hist, x_ref)
        if ok:
            a_ctrl = self.ctrl.update(ci)
        else:
            if self.sup.t_trip == self.t:
                self.events.append(f"t={self.t:.3f}s SAFETY STOP: {self.sup.reason}")
            a_ctrl = Supervisor.stop_command(self.integrator.v_cmd, c.safety.a_brake, c.Ts)
        v_send, a_eff = self.integrator.step(a_ctrl)
        self.x_dr += v_send * c.Ts
        self.u_hist.appendleft(a_eff)
        self.a_prev = a_eff
        jitter = abs(self.rng.normal(0.0, c.driver.jitter)) if c.driver.jitter > 0 else 0.0
        self.cmd_queue.append((self.t + c.driver.latency + jitter, v_send))

        # ---- log (state at t_k)
        L = self.log
        for k, v in (
            ("t", self.t), ("x", self.x_c), ("v", self.v_c), ("theta", self.theta), ("omega", self.omega),
            ("adc", adc), ("theta_meas", theta_meas), ("x_meas", x_meas),
            ("x_hat", est.x), ("v_hat", est.v), ("theta_hat", est.theta), ("omega_hat", est.omega), ("bias_hat", est.bias),
            ("a_ctrl", a_ctrl), ("a_eff", a_eff), ("v_cmd", self.integrator.v_cmd), ("v_motor", self.v_m),
            ("x_ref", x_ref), ("tripped", int(self.sup.tripped)),
            ("omega0_hat", self.adapter.rls.omega0 if self.adapter is not None else float("nan")),
        ):
            L[k].append(v)

        # ---- integrate plant over one period
        dt = c.Ts / c.substeps
        for _ in range(c.substeps):
            while self.cmd_queue and self.cmd_queue[0][0] <= self.t + 1e-12:
                self.v_target = self.cmd_queue.popleft()[1]
            self._substep(dt)
            self.t += dt
            self.hist.append((self.t, self.theta, self.x_m))
        self.k += 1
        self.t = self.k * c.Ts  # kill float drift

        if abs(self.theta) > math.radians(100) and not c.safety.theta_trip > math.pi:
            self.events.append(f"t={self.t:.2f}s rod fell")
            self.finished = True
        if self.sup.tripped and self.t - self.sup.t_trip > c.end_after_trip:
            self.finished = True
        if self.t >= c.duration - 1e-9:
            self.finished = True
        return not self.finished

    def _substep(self, dt: float) -> None:
        c, p, drv = self.cfg, self.p, self.cfg.driver
        # driver velocity loop (first-order lag + acceleration limit), exact for the lag
        if drv.tau > 0:
            dv = (self.v_target - self.v_m) * (1.0 - math.exp(-dt / drv.tau))
        else:
            dv = self.v_target - self.v_m
        dv = max(-drv.accel_limit * dt, min(drv.accel_limit * dt, dv))
        self.v_m += dv
        self.x_m += self.v_m * dt
        # belt backlash (play operator): cart follows the motor only at the ends of the gap
        v_old = self.v_c
        if drv.backlash > 0:
            half = drv.backlash / 2.0
            x_free = self.x_c + self.v_c * dt
            rel = x_free - self.x_m
            if rel > half:
                x_new, self.v_c = self.x_m + half, self.v_m
            elif rel < -half:
                x_new, self.v_c = self.x_m - half, self.v_m
            else:
                x_new = x_free
            a_cart = (self.v_c - v_old) / dt
            self.theta, self.omega = rk4_pendulum(p, self.theta, self.omega, a_cart, dt)
            self.x_c = x_new
        else:
            self.v_c = self.v_m
            a_cart = (self.v_c - v_old) / dt
            self.theta, self.omega = rk4_pendulum(p, self.theta, self.omega, a_cart, dt)
            self.x_c = self.x_m
        # hard rail ends
        if abs(self.x_c) > c.rail_half:
            self.x_c = math.copysign(c.rail_half, self.x_c)
            self.x_m = self.x_c
            dvc = -self.v_c
            self.omega += -p.beta * math.cos(self.theta) * dvc  # impulse from the crash
            self.v_c = self.v_m = 0.0
            if not self.collided:
                self.collided = True
                self.events.append(f"t={self.t:.2f}s CART HIT RAIL END")

    # ------------------------------------------------------------------- run
    def run(self) -> "SimResult":
        while self.step():
            pass
        return self.result()

    def result(self) -> "SimResult":
        return SimResult({k: np.asarray(v, float) for k, v in self.log.items()}, list(self.events),
                         self.sup.reason, self.collided, self.ctrl.describe())


@dataclass
class SimResult:
    data: dict[str, np.ndarray]
    events: list[str]
    trip_reason: str
    collided: bool
    controller: dict

    def __getitem__(self, k: str) -> np.ndarray:
        return self.data[k]

    @property
    def balanced(self) -> bool:
        return not self.trip_reason and not self.collided

    def metrics(self, settle_band_deg: float = 1.0) -> dict:
        t, th, x, a = self["t"], self["theta"], self["x"], self["a_eff"]
        if len(t) == 0:
            return {}
        band = math.radians(settle_band_deg)
        outside = np.nonzero(np.abs(th) > band)[0]
        settle = float(t[outside[-1] + 1]) if len(outside) and outside[-1] + 1 < len(t) else (0.0 if not len(outside) else float("nan"))
        tail = t > t[-1] - 2.0
        return {
            "balanced": self.balanced,
            "settling_time_theta_s": settle,
            "max_abs_theta_deg": float(np.degrees(np.max(np.abs(th)))),
            "max_abs_x_cm": float(np.max(np.abs(x)) * 100),
            "rms_theta_last2s_deg": float(np.degrees(np.sqrt(np.mean(th[tail] ** 2)))),
            "rms_x_last2s_cm": float(np.sqrt(np.mean(x[tail] ** 2)) * 100),
            "max_abs_accel": float(np.max(np.abs(a))),
            "rms_accel": float(np.sqrt(np.mean(a**2))),
            "trip_reason": self.trip_reason,
        }

    def to_csv(self, path: str | Path) -> None:
        keys = list(self.data.keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for row in zip(*(self.data[k] for k in keys)):
                w.writerow([f"{v:.6g}" for v in row])


def wrapped(theta: np.ndarray) -> np.ndarray:
    return (np.asarray(theta) + math.pi) % (2 * math.pi) - math.pi


__all__ = ["Simulation", "SimConfig", "SimResult", "DriverModel", "SensorModel", "Calibration", "Disturbance", "wrap_angle"]
