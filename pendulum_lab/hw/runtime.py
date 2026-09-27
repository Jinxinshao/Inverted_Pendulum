"""Real-time loop on the physical rig.

Modes
-----
* ``shadow``  - reads the real angle sensor, runs estimator + controller,
                but NEVER sends motion. The estimator is told the truth (cart
                at rest at x = 0), so the logged a_ctrl is exactly what the
                controller WOULD command for the rod angle you set by hand.
                Use it to check signs (lean the rod to +x -> a_ctrl > 0),
                noise and timing safely.
* ``closed``  - sends the velocity command to the PD42S1 every period.

Sequence: PREFLIGHT -> WAIT_ARM (rod held within +-arm_deg for arm_time)
          -> ARMED ("release now"; motor still at zero)
          -> ACTIVE as soon as the rod moves freely (> release_deg from the held angle)
          -> STOPPING (ramp to zero) -> DONE
Why release-triggered: if control starts while a hand still holds the rod, the
cart accelerates under a rod that cannot respond, and the Kalman filter learns
that as a zero bias; measured on the fake rig, holding 0.2 s after a timed start
made LQR and PID hit the soft limit (docs/05 step 7).
Any exception, Ctrl+C or GUI stop goes to STOPPING; if the zero-speed
command cannot be delivered the operator is told to cut the 12 V supply.
"""
from __future__ import annotations

import csv
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..config import make_controller, make_estimator, plant_from, safety_from, actuator_from
from ..control.actuator import VelocityCommandIntegrator
from ..control.controllers import ControlInput
from ..control.safety import Supervisor
from .motor import MotorWorker
from .sensor import AngleCalibration, SensorReader
from .timing import HighResTimer, PeriodicTimer

LOG_COLUMNS = (
    "t", "dt", "state", "seq", "adc", "sensor_age", "theta_meas", "x_meas",
    "x_hat", "v_hat", "theta_hat", "omega_hat", "bias_hat",
    "a_ctrl", "a_eff", "v_cmd", "motor_err", "tripped", "omega0_hat",
)


@dataclass
class RunOptions:
    algorithm: str = "lqr"
    mode: str = "shadow"          # 'shadow' | 'closed'
    duration: float = 60.0        # s of ACTIVE control
    arm_deg: float = 3.0
    arm_time: float = 1.0
    arm_mode: str = "release"     # 'release': start when the rod moves freely; 'timer': start right after arm_time
    release_deg: float = 0.4      # deviation from the held angle that counts as "released" (3 samples, same sign)
    release_timeout: float = 5.0
    x_ref: float = 0.0
    min_sensor_rate_ratio: float = 0.95   # sensor rate must be >= this * 1/Ts
    log_path: str | None = None
    adapt: str = "monitor"        # online omega0 identification: 'off' | 'monitor' | 'update'
    excitation_amp: float = 0.0   # m, known x_ref square wave for identification (0 = none)


@dataclass
class LiveState:
    """Thread-safe snapshot for the GUI."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    rows: deque = field(default_factory=lambda: deque(maxlen=4000))
    state: str = "INIT"
    message: str = ""


def calibration_from(hw: dict) -> AngleCalibration:
    c = hw["calibration"]
    missing = [k for k in ("adc_upright", "counts_per_rad", "sign") if c.get(k) is None]
    if missing:
        raise ValueError(f"hardware.calibration not measured yet: {missing} (see docs/05 step 2)")
    return AngleCalibration(float(c["adc_upright"]), float(c["counts_per_rad"]), int(c["sign"]))


class HardwareLoop:
    def __init__(self, cfg: dict, sensor: SensorReader, motor: MotorWorker | None, opts: RunOptions,
                 calib: AngleCalibration, live: LiveState | None = None, printer=print):
        if opts.mode == "closed" and motor is None:
            raise ValueError("closed mode needs a motor")
        self.cfg, self.sensor, self.motor, self.opts, self.calib = cfg, sensor, motor, opts, calib
        self.p = plant_from(cfg)
        self.Ts = cfg["loop"]["Ts"]
        self.ctrl = make_controller(opts.algorithm, self.p, cfg)
        self.est = make_estimator(self.p, cfg, opts.algorithm)
        lim = safety_from(cfg)
        if opts.mode == "shadow":
            # the real cart does not move: only sensor health can end a shadow run
            lim = replace(lim, theta_trip=1e9, x_soft=1e9, check_adc_range=False)
        self.sup = Supervisor(lim)
        self.integ = VelocityCommandIntegrator(self.Ts, actuator_from(cfg))
        self.live = live or LiveState()
        self.print = printer
        self.stop_request = threading.Event()
        self.rows: list[tuple] = []
        self.result_message = ""
        self.adapter = None
        if opts.adapt != "off" and opts.mode == "closed":
            from ..control.online_id import AdaptiveUpdater

            self.adapter = AdaptiveUpdater(self.p, cfg, opts.algorithm, opts.adapt,
                                           excitation_amp=opts.excitation_amp)

    @property
    def adapt_log(self) -> list:
        return self.adapter.history if self.adapter else []

    # ----------------------------------------------------------- preflight
    def preflight(self, wait: float = 1.0) -> dict:
        t_end = time.perf_counter() + wait
        seq0 = self.sensor.seq
        while time.perf_counter() < t_end:
            time.sleep(0.01)
        n = self.sensor.seq - seq0
        rate = n / wait
        need = self.opts.min_sensor_rate_ratio / self.Ts
        s = self.sensor.latest()
        info = {"sensor_rate_hz": rate, "required_hz": need, "adc": s.adc if s else None}
        if rate < need:
            raise RuntimeError(
                f"angle sensor streams {rate:.0f} Hz but the loop needs >= {need:.0f} Hz (Ts={self.Ts * 1e3:.1f} ms). "
                "Raise the firmware output rate (docs/05 step 1) or increase loop.Ts and redesign."
            )
        if self.opts.mode == "closed":
            drv = self.motor.drv
            # the manual's usage rule: clear state -> enable -> zero speed before any motion
            # (a disabled motor, e.g. after drivertest or after a brake, used to fail here)
            done = drv.prepare_for_motion()
            self.print("driver: " + " -> ".join(done))
            ok, problems, dinfo = drv.readiness()
            info["driver"] = dinfo
            if not ok:
                raise RuntimeError("PD42S1 not ready: " + "; ".join(problems))
            cart = self.cfg["hardware"]["cart"]
            if cart.get("center_mode", "start") == "start" or cart.get("center_counts") is None:
                # the rail centre is where the operator put the cart before pressing start;
                # the soft limits are +-x_soft around it
                c = drv.read_position()
                self.motor.units.center_counts = float(c)
                info["center_counts"] = c
                self.print(f"rail centre = start position (encoder {c}); soft limits +-{self.sup.lim.x_soft * 100:.0f} cm")
            # cart must start well inside the soft limits
            self.motor.read_position_now()
            if self.motor.position is None:
                raise RuntimeError(f"cannot read motor position: {self.motor.last_error}")
            x0 = self.motor.position[1]
            if abs(x0) > 0.5 * self.sup.lim.x_soft:
                raise RuntimeError(f"cart at {x0 * 100:+.1f} cm: move it to the rail centre first")
            info["x0"] = x0
        return info

    def request_stop(self) -> None:
        self.stop_request.set()

    def _set_state(self, st: str, msg: str = "") -> None:
        with self.live.lock:
            self.live.state = st
            if msg:
                self.live.message = msg
        if msg:
            self.print(f"[{st}] {msg}")

    # ----------------------------------------------------------------- run
    def run(self) -> str:
        o, Ts = self.opts, self.Ts
        timer = PeriodicTimer(Ts)
        self.overruns = 0
        try:
            self._set_state("PREFLIGHT", "checking sensor stream")
            info = self.preflight()
        except Exception as e:  # noqa: BLE001
            self.result_message = f"PREFLIGHT FAILED: {e}"
            self._final_stop()
            self._set_state("DONE", self.result_message)
            return self.result_message
        self.print(f"sensor {info['sensor_rate_hz']:.0f} Hz (need {info['required_hz']:.0f}), ADC={info['adc']}")
        state = "WAIT_ARM"
        self._set_state(state, f"hold the rod within +-{o.arm_deg:.0f} deg of upright for {o.arm_time:.1f} s")
        arm_since = None
        hold_buf: deque = deque(maxlen=max(3, int(0.3 / Ts)))
        theta_hold = armed_at = prev_dev = 0.0
        dev_run = 0
        t_active = None
        x_virtual = 0.0
        a_prev = 0.0
        u_hist: deque = deque([0.0] * 32, maxlen=32)
        timer = PeriodicTimer(Ts)
        t_prev = time.perf_counter()
        try:
            with HighResTimer():
                while True:
                    now = timer.wait()
                    dt, t_prev = now - t_prev, now
                    s = self.sensor.latest()
                    if s is None or not self.sensor.running:
                        raise RuntimeError(self.sensor.error or "no angle samples")
                    age = now - s.t
                    theta_m = self.calib.theta(s.adc)

                    # cart position: encoder if available, else dead reckoning of commands
                    if (self.motor is not None and o.mode == "closed" and self.motor.feedback_every
                            and self.motor.position is not None):
                        x_m = self.motor.position[1]
                    else:
                        x_m = x_virtual

                    if state in ("WAIT_ARM", "ARMED"):
                        if self.stop_request.is_set():
                            state = "DONE"
                            break
                        inside = abs(theta_m) < math.radians(o.arm_deg)
                        go = False
                        if state == "WAIT_ARM":
                            if inside:
                                arm_since = arm_since or now
                                hold_buf.append(theta_m)
                                if now - arm_since >= o.arm_time:
                                    if o.arm_mode == "timer" or o.mode == "shadow":  # shadow: nothing moves
                                        go = True
                                    else:
                                        state = "ARMED"
                                        theta_hold = sum(hold_buf) / len(hold_buf)
                                        armed_at = now
                                        dev_run = 0
                                        self._set_state(state, "RELEASE the rod now - control starts when it moves freely")
                            else:
                                arm_since = None
                                hold_buf.clear()
                        else:  # ARMED: start as soon as the rod is FREE (a held rod would fool the estimator)
                            dev = theta_m - theta_hold
                            if abs(dev) > math.radians(o.release_deg):
                                dev_run = dev_run + 1 if (dev_run == 0 or (dev > 0) == (prev_dev > 0)) else 1
                                prev_dev = dev
                            else:
                                dev_run = 0
                            if dev_run >= 3 or not inside:
                                go = inside
                                if not inside:  # moved out of the window before we started: re-arm
                                    state, arm_since = "WAIT_ARM", None
                                    hold_buf.clear()
                                    self._set_state(state, "rod left the window - hold it upright again")
                            elif now - armed_at > o.release_timeout:
                                state, arm_since = "WAIT_ARM", None
                                hold_buf.clear()
                                self._set_state(state, "no release detected - hold the rod upright again")
                        if go:
                            state = "ACTIVE"
                            t_active = now
                            self.est.reset(theta_m, x_m)
                            self.ctrl.reset()
                            self.integ.reset(0.0)
                            x_virtual = 0.0
                            self._set_state(state, f"control ON ({o.algorithm}, {o.mode})")
                        self._log(now, dt, state, s, age, theta_m, x_m, None, 0.0, 0.0)
                        continue

                    shadow = o.mode == "shadow"
                    if shadow:
                        # honest state: the real cart stands still at x = 0, a = 0
                        x_m = 0.0
                        self.integ.reset(0.0)
                        a_prev = 0.0
                    est = self.est.update(theta_m, x_m, self.integ.v_cmd, a_prev)
                    runtime = now - t_active
                    motor_err = self.motor.consecutive_errors if (self.motor and o.mode == "closed") else 0
                    ok = self.sup.check(runtime, est.x, est.v, est.theta, sensor_age=age, motor_errors=motor_err,
                                        adc=s.adc, runtime=runtime)
                    if state == "ACTIVE" and (not ok or self.stop_request.is_set() or runtime > o.duration):
                        state = "STOPPING"
                        why = self.sup.reason or ("stop requested" if self.stop_request.is_set() else "duration reached")
                        self.result_message = why
                        self._set_state(state, why)
                    if state == "ACTIVE" and self.adapter is not None:
                        new_ctrl = self.adapter.step(runtime, theta_m - est.bias, a_prev, self.ctrl, self.est)
                        if new_ctrl is not None:
                            self.ctrl = new_ctrl
                            self.print(f"controller redesigned: omega0 = {self.adapter.p_design.omega0:.4f} rad/s")
                    if state == "ACTIVE":
                        x_ref = o.x_ref + (self.adapter.reference(runtime) if self.adapter is not None else 0.0)
                        ci = ControlInput(runtime, est.x, est.v, est.theta, est.omega, self.integ.v_cmd, u_hist, x_ref)
                        a = self.ctrl.update(ci)
                    else:
                        a = Supervisor.stop_command(self.integ.v_cmd, self.sup.lim.a_brake, Ts)
                    v_send, a_eff = self.integ.step(a)
                    if shadow:
                        v_send = 0.0  # nothing is sent; a_ctrl in the log is what WOULD be commanded
                        a_eff = 0.0
                    u_hist.appendleft(a_eff)
                    a_prev = a_eff
                    x_virtual += v_send * Ts  # (unused in closed mode with encoder feedback)
                    if o.mode == "closed":
                        self.motor.set_velocity(v_send)
                    self._log(now, dt, state, s, age, theta_m, x_m, est, a, a_eff)
                    if state == "STOPPING" and (shadow or self.integ.v_cmd == 0.0):
                        break
        except KeyboardInterrupt:
            self.result_message = "Ctrl+C"
        except Exception as e:  # noqa: BLE001
            self.result_message = f"ERROR: {e}"
        finally:
            self._final_stop()
            self._set_state("DONE", self.result_message or "finished")
            self.overruns = timer.overruns
            if o.log_path:
                self.save(o.log_path)
        return self.result_message

    def _final_stop(self) -> None:
        if self.motor is not None and self.opts.mode == "closed":
            if not self.motor.stop_motion():
                # zero speed not confirmed: try the latching FC brake (official emergency stop)
                if self.motor.emergency_brake():
                    self.print("zero-speed not acknowledged -> FC BRAKE sent (use 清除电机状态 + 使能 to recover)")
                else:
                    self.print("!!! DRIVER NOT ANSWERING - SWITCH OFF THE 12 V SUPPLY NOW !!!")

    # ----------------------------------------------------------------- log
    def _log(self, now, dt, state, s, age, theta_m, x_m, est, a, a_eff):
        motor_err = self.motor.consecutive_errors if self.motor else 0
        row = (
            now, dt, state, s.seq, s.adc, age, theta_m, x_m,
            est.x if est else float("nan"), est.v if est else float("nan"),
            est.theta if est else float("nan"), est.omega if est else float("nan"), est.bias if est else float("nan"),
            a, a_eff, self.integ.v_cmd, motor_err, int(self.sup.tripped),
            self.adapter.rls.omega0 if self.adapter else float("nan"),
        )
        self.rows.append(row)
        with self.live.lock:
            self.live.rows.append(row)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        t0 = self.rows[0][0] if self.rows else 0.0
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(LOG_COLUMNS)
            for r in self.rows:
                w.writerow([f"{r[0] - t0:.6f}"] + [x if isinstance(x, str) else f"{x:.6g}" for x in r[1:]])
        self.print(f"log written: {path} ({len(self.rows)} rows)")
