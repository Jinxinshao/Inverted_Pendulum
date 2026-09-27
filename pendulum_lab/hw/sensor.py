"""Angle sensor (potentiometer + ADC board streaming text over USB serial).

Accepted line formats (first match wins):
    "ADC=1966, 1584 mV"      <- current firmware (VOFA+ screenshot)
    "ADC=1966,T=123456"      <- recommended firmware (adds MCU time stamp in us)
    "1966"                   <- bare integer
A sample is time-stamped with the PC's perf_counter when its line arrives.
If the firmware provides T=..., the MCU time stamp is kept too so the true
sampling jitter can be separated from USB/OS jitter.
"""
from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

_RE_ADC = re.compile(rb"ADC\s*[=:]\s*(-?\d+)", re.I)
_RE_T = re.compile(rb"\bT\s*[=:]\s*(\d+)", re.I)
_RE_INT = re.compile(rb"^\s*(-?\d+)")


def parse_line(line: bytes) -> tuple[int | None, int | None]:
    """Return (adc, mcu_time_us) or (None, None) if the line carries no reading."""
    m = _RE_ADC.search(line)
    if not m:
        m = _RE_INT.match(line)
    if not m:
        return None, None
    t = _RE_T.search(line)
    return int(m.group(1)), (int(t.group(1)) if t else None)


@dataclass
class AngleCalibration:
    """theta = sign * (adc - adc_upright) / counts_per_rad.

    Defaults for this rig:
      * counts_per_rad = 678.1 +- 0.8 from the first free-swing recording
        (2026-09-27, period-amplitude relation, docs/02 section 2.10). The old
        gravity value (4092 - 1966)/pi = 676.7 used the HANGING reading, which is
        clipped on this rig: the true bottom lies 0.5 deg inside the dead zone
        (reading 4098 by extrapolation); (4098.3 - 1966)/pi = 678.7 agrees with
        the swing value. Datasheet (345 deg over 4095 counts): 680.1.
      * adc_upright = 1966 from the hand-balanced reading; refine with the
        calibration procedure (docs/05) and the Kalman bias estimate.
      * sign must be checked on the rig (tilt the rod toward the motor's
        positive direction: theta must become positive).
    """

    adc_upright: float = 1966.0
    counts_per_rad: float = 678.1
    sign: int = 1

    def theta(self, adc: float) -> float:
        return self.sign * (adc - self.adc_upright) / self.counts_per_rad

    @property
    def deg_per_count(self) -> float:
        return math.degrees(1.0 / self.counts_per_rad)

    @classmethod
    def from_upright_and_hanging(cls, adc_up: float, adc_down: float, sign: int = 1) -> "AngleCalibration":
        return cls(adc_up, abs(adc_down - adc_up) / math.pi, sign)


@dataclass
class Sample:
    t: float            # PC time (perf_counter) of arrival
    adc: int
    mcu_t: int | None
    seq: int


class SensorReader(threading.Thread):
    """Background reader keeping the latest sample and statistics."""

    def __init__(self, port, keep_seconds: float = 600.0, expected_rate: float = 500.0):
        super().__init__(daemon=True, name="angle-sensor")
        self.port = port
        self.lock = threading.Lock()
        self._latest: Sample | None = None
        self.seq = 0
        self.parse_errors = 0
        self.lines = 0
        self.history: deque = deque(maxlen=int(keep_seconds * expected_rate))
        self.running = True
        self.error: str = ""
        self._buf = bytearray()

    def run(self) -> None:
        while self.running:
            try:
                n = getattr(self.port, "in_waiting", 0) or 1
                chunk = self.port.read(n)
            except Exception as e:  # noqa: BLE001  (USB unplugged etc.)
                self.error = f"serial read failed: {e}"
                self.running = False
                break
            if not chunk:
                continue
            now = time.perf_counter()
            self._buf.extend(chunk)
            while True:
                i = self._buf.find(b"\n")
                if i < 0:
                    break
                line = bytes(self._buf[:i])
                del self._buf[: i + 1]
                self.lines += 1
                adc, mt = parse_line(line)
                if adc is None:
                    self.parse_errors += 1
                    continue
                self.seq += 1
                s = Sample(now, adc, mt, self.seq)
                with self.lock:
                    self._latest = s
                    self.history.append((now, adc, mt))
            if len(self._buf) > 4096:  # no newline for a long time: garbage
                self._buf.clear()
                self.parse_errors += 1

    def latest(self) -> Sample | None:
        with self.lock:
            return self._latest

    def stop(self) -> None:
        self.running = False

    def snapshot(self) -> list[tuple[float, int, int | None]]:
        with self.lock:
            return list(self.history)


def sensor_statistics(samples: list[tuple[float, int, int | None]]) -> dict:
    """Rate, jitter and noise statistics of a list of (t, adc, mcu_t)."""
    import numpy as np

    if len(samples) < 3:
        return {"n": len(samples)}
    t = np.array([s[0] for s in samples])
    a = np.array([s[1] for s in samples], float)
    dt = np.diff(t)
    out = {
        "n": len(samples),
        "duration_s": float(t[-1] - t[0]),
        "rate_hz": float((len(t) - 1) / (t[-1] - t[0])) if t[-1] > t[0] else float("nan"),
        "dt_median_ms": float(np.median(dt) * 1e3),
        "dt_p99_ms": float(np.percentile(dt, 99) * 1e3),
        "dt_max_ms": float(np.max(dt) * 1e3),
        "adc_mean": float(np.mean(a)),
        "adc_std": float(np.std(a)),
        "adc_min": int(np.min(a)),
        "adc_max": int(np.max(a)),
        "zero_dt_fraction": float(np.mean(dt < 1e-4)),
    }
    mt = [s[2] for s in samples if s[2] is not None]
    if len(mt) > 2:
        d = np.diff(np.array(mt, float)) * 1e-3
        out["mcu_dt_median_ms"] = float(np.median(d))
        out["mcu_dt_max_ms"] = float(np.max(d))
    return out
