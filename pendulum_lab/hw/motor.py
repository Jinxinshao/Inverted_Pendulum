"""Cart motion through the PD42S1 in communication speed mode (F1).

Unit model (one calibration number):
    l = belt travel per motor revolution [m/rev]   (GT2 20T pulley: 0.040 m)
    rpm        = v [m/s] * 60 / l        (+x is DEFINED as the 正转 direction)
    x [m]      = sign * (counts - center_counts) / 51200 * l
``sign`` is the encoder direction: +1 if the position counter increases
while the motor turns 正转 (measured by calibrate-cart).

A dedicated thread owns the motor serial port so the control loop never
blocks on a slow reply: the loop posts the newest velocity command, the
thread sends it (plus a periodic position read) and publishes the results.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .pd42s1 import COUNTS_PER_REV, PD42S1


@dataclass
class CartUnits:
    meters_per_rev: float          # belt travel per motor turn [m]
    center_counts: float = 0.0     # encoder reading at the rail centre
    sign: int = 1                  # encoder direction: +1 if counts increase for 正转

    def x_of_counts(self, counts: float) -> float:
        return self.sign * (counts - self.center_counts) / COUNTS_PER_REV * self.meters_per_rev

    def rpm_of(self, v: float) -> float:
        return v * 60.0 / self.meters_per_rev

    @property
    def counts_per_meter(self) -> float:
        return COUNTS_PER_REV / self.meters_per_rev

    @classmethod
    def from_config(cls, hw: dict) -> "CartUnits":
        c = hw["cart"]
        if c.get("meters_per_rev") is None:
            raise ValueError("hardware.cart.meters_per_rev not measured yet (docs/05 step 4: calibrate-cart)")
        return cls(float(c["meters_per_rev"]), float(c.get("center_counts") or 0.0), int(c.get("sign") or 1))


class MotorWorker(threading.Thread):
    def __init__(self, drv: PD42S1, units: CartUnits, feedback_every: int = 2, accel: int = 0, keepalive: float = 0.05):
        super().__init__(daemon=True, name="pd42s1")
        self.drv, self.units = drv, units
        self.feedback_every = feedback_every
        self.accel = accel                 # F1 accel byte (0 = follow command immediately)
        self.keepalive = keepalive
        self.cv = threading.Condition()
        self._v_req = 0.0
        self._new = False
        self.running = True
        self.paused = False
        self.consecutive_errors = 0
        self.total_errors = 0
        self.n_sent = 0
        self.position: tuple[float, float] | None = None  # (pc time, x [m])
        self.last_error = ""
        self.cmd_latency: list[float] = []

    # ---------------------------------------------------------- API (loop)
    def set_velocity(self, v: float) -> None:
        with self.cv:
            self._v_req = v
            self._new = True
            self.cv.notify()

    def stop(self) -> None:
        self.running = False
        with self.cv:
            self.cv.notify()

    # ---------------------------------------------------------- thread
    def _send(self, v: float) -> None:
        t0 = time.perf_counter()
        try:
            self.drv.set_speed(self.units.rpm_of(v), self.accel)
            self.consecutive_errors = 0
            self.cmd_latency.append(time.perf_counter() - t0)
            if len(self.cmd_latency) > 100000:
                del self.cmd_latency[:50000]
        except Exception as e:  # noqa: BLE001
            self.consecutive_errors += 1
            self.total_errors += 1
            self.last_error = str(e)
        self.n_sent += 1

    def read_position_now(self) -> None:
        t0 = time.perf_counter()
        try:
            c = self.drv.read_position()
            t1 = time.perf_counter()
            self.position = (0.5 * (t0 + t1), self.units.x_of_counts(c))
            self.consecutive_errors = 0
        except Exception as e:  # noqa: BLE001
            self.consecutive_errors += 1
            self.total_errors += 1
            self.last_error = str(e)

    def run(self) -> None:
        k = 0
        while self.running:
            with self.cv:
                if not self._new:
                    self.cv.wait(self.keepalive)
                v, self._new = self._v_req, False
            if not self.running:
                break
            if self.paused:
                continue
            self._send(v)
            k += 1
            if self.feedback_every and k % self.feedback_every == 0:
                self.read_position_now()

    # ---------------------------------------------------------- stopping
    def stop_motion(self, retries: int = 5) -> bool:
        """Zero speed directly (bypassing the queue). True if acknowledged."""
        self.set_velocity(0.0)
        for _ in range(retries):
            try:
                self.drv.set_speed(0.0, self.accel)
                return True
            except Exception as e:  # noqa: BLE001
                self.last_error = str(e)
                time.sleep(0.01)
        return False

    def emergency_brake(self) -> bool:
        """FC brake (latches; recover with clear_state + enable)."""
        self.paused = True
        for _ in range(3):
            try:
                self.drv.brake()
                return True
            except Exception as e:  # noqa: BLE001
                self.last_error = str(e)
        return False


def open_driver(hw: dict, port_name: str | None = None, timeout: float = 0.05, log=None):
    """Open the motor serial port from the 'hardware' config section. Returns (driver, port)."""
    from ..tools.hwtools import open_serial

    port = open_serial(port_name or hw["motor_port"], hw["motor_baud"])
    return PD42S1(port, hw.get("motor_protocol", "custom"), hw["motor_address"], timeout=timeout, log=log), port


def make_worker(drv: PD42S1, hw: dict, units: CartUnits | None = None) -> MotorWorker:
    return MotorWorker(drv, units or CartUnits.from_config(hw), hw.get("motor_feedback_every", 2), hw.get("motor_accel", 0))


def fake_units() -> CartUnits:
    """Units matching FakeRig defaults (40 mm/rev, centre 100000 counts)."""
    return CartUnits(0.040, 100000.0, 1)
