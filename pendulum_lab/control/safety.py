"""Safety supervisor shared by the simulator and the hardware runtime.

The same code trips in simulation and on the rig, so students see exactly
which condition would have stopped the real cart.

Stop logic ("controlled stop"): once tripped, the velocity command ramps to
zero at ``a_brake`` and stays zero. The PD42S1 "FC brake" is NOT used for
routine stops because it latches (see docs/07).

Stopping-distance prediction: with latency tau and braking deceleration a_b,
a cart at x moving with v ends at
    x_stop = x + v tau + v|v| / (2 a_b)
and the supervisor trips when |x_stop| exceeds the soft limit, i.e. BEFORE the
cart gets there.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class SafetyLimits:
    theta_trip: float = math.radians(25.0)   # rod considered fallen / unrecoverable
    x_soft: float = 0.22                     # m from rail centre (rail ~ +-0.35 m, cart width!)
    a_brake: float = 3.0                     # m/s^2 used for stopping
    latency: float = 0.03                    # s, assumed for stopping distance
    v_max: float = 0.6                       # m/s
    sensor_timeout: float = 0.05             # s without a fresh angle sample
    max_motor_errors: int = 3                # consecutive failed motor transactions
    max_runtime: float = 300.0               # s
    adc_min_valid: int = 20                  # potentiometer near its end / dead zone
    adc_max_valid: int = 4075
    check_adc_range: bool = True             # False for swing-up (passes the dead zone on purpose)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["theta_trip_deg"] = math.degrees(self.theta_trip)
        return d


class Supervisor:
    def __init__(self, limits: SafetyLimits):
        self.lim = limits
        self.tripped = False
        self.reason = ""
        self.t_trip = None

    def reset(self) -> None:
        self.tripped = False
        self.reason = ""
        self.t_trip = None

    def stopping_position(self, x: float, v: float) -> float:
        L = self.lim
        return x + v * L.latency + v * abs(v) / (2.0 * L.a_brake)

    def check(self, t: float, x: float, v: float, theta: float, sensor_age: float = 0.0,
              motor_errors: int = 0, adc: int | None = None, runtime: float = 0.0) -> bool:
        """Return True if OK. Latches the first failure reason."""
        if self.tripped:
            return False
        L = self.lim
        reason = ""
        if abs(theta) > L.theta_trip:
            reason = f"angle {math.degrees(theta):+.1f} deg beyond +-{math.degrees(L.theta_trip):.0f} deg"
        elif abs(x) > L.x_soft:
            reason = f"cart at {x * 100:+.1f} cm beyond soft limit +-{L.x_soft * 100:.0f} cm"
        elif abs(self.stopping_position(x, v)) > L.x_soft + 0.02:
            reason = f"predicted stop at {self.stopping_position(x, v) * 100:+.1f} cm (x={x*100:+.1f} cm, v={v:+.2f} m/s)"
        elif sensor_age > L.sensor_timeout:
            reason = f"angle sensor stale for {sensor_age * 1e3:.0f} ms"
        elif motor_errors >= L.max_motor_errors:
            reason = f"{motor_errors} consecutive motor communication errors"
        elif L.check_adc_range and adc is not None and not (L.adc_min_valid <= adc <= L.adc_max_valid):
            reason = f"ADC {adc} at potentiometer end / dead zone"
        elif runtime > L.max_runtime:
            reason = f"run time {runtime:.0f} s exceeded"
        if reason:
            self.tripped, self.reason, self.t_trip = True, reason, t
            return False
        return True

    @staticmethod
    def stop_command(v_cmd: float, a_brake: float, Ts: float) -> float:
        """Acceleration that ramps the velocity command to zero without overshoot."""
        if abs(v_cmd) <= a_brake * Ts:
            return -v_cmd / Ts
        return -math.copysign(a_brake, v_cmd)
