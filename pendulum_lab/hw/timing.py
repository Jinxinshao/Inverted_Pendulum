"""Fixed-period scheduling on a desktop OS.

Windows' default timer granularity is 15.6 ms, which alone would make a 5 ms
loop impossible. We (1) request 1 ms resolution via winmm.timeBeginPeriod,
(2) sleep until ~1.5 ms before the deadline and (3) spin for the rest.
Deadlines are absolute (k * Ts from the start), so errors do not accumulate;
an overrun skips to the next future deadline and is counted.
Python >= 3.11 on Windows already uses high-resolution waitable timers for
time.sleep, which helps further.
"""
from __future__ import annotations

import sys
import time


class HighResTimer:
    def __enter__(self):
        self._win = sys.platform.startswith("win")
        if self._win:
            try:
                import ctypes

                ctypes.windll.winmm.timeBeginPeriod(1)
            except Exception:  # noqa: BLE001
                self._win = False
        return self

    def __exit__(self, *exc):
        if self._win:
            import ctypes

            ctypes.windll.winmm.timeEndPeriod(1)
        return False


class PeriodicTimer:
    def __init__(self, period: float, spin: float = 0.0015):
        self.period = period
        self.spin = spin
        self.t0 = time.perf_counter()
        self.k = 0
        self.overruns = 0

    def wait(self) -> float:
        """Block until the next deadline; return the actual wake-up time."""
        self.k += 1
        deadline = self.t0 + self.k * self.period
        now = time.perf_counter()
        if now > deadline + self.period:  # overrun: resynchronise
            self.overruns += 1
            self.k = int((now - self.t0) / self.period) + 1
            deadline = self.t0 + self.k * self.period
        rem = deadline - now - self.spin
        if rem > 0:
            time.sleep(rem)
        while time.perf_counter() < deadline:
            pass
        return time.perf_counter()
