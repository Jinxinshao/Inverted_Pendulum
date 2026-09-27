"""Host-side actuator interface: acceleration command -> velocity command.

Identical in simulation and on hardware. The PD42S1's own closed velocity loop
is the "inner loop"; the PC only decides how that velocity command evolves.

    v_cmd[k+1] = clip(v_cmd[k] + clip(a_cmd, +-a_max) * Ts, +-v_max)

The EFFECTIVE acceleration (v_cmd[k+1] - v_cmd[k]) / Ts is what the plant
receives, so it - not the raw controller output - is fed to the estimator
and to the delayed-input history of the LQR (built-in anti-windup).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActuatorLimits:
    a_max: float = 6.0     # m/s^2, command clip
    v_max: float = 0.6     # m/s, command clip
    v_quantum: float = 0.0  # m/s per LSB of the driver speed command (0 = continuous)


class VelocityCommandIntegrator:
    def __init__(self, Ts: float, limits: ActuatorLimits):
        self.Ts = Ts
        self.lim = limits
        self.v_cmd = 0.0          # exact integrator state
        self.v_sent = 0.0         # quantised value actually sent

    def reset(self, v0: float = 0.0) -> None:
        self.v_cmd = v0
        self.v_sent = v0

    def step(self, a_cmd: float) -> tuple[float, float]:
        """Return (velocity command to send, effective acceleration)."""
        L = self.lim
        a = max(-L.a_max, min(L.a_max, a_cmd))
        v_new = max(-L.v_max, min(L.v_max, self.v_cmd + a * self.Ts))
        a_eff = (v_new - self.v_cmd) / self.Ts
        self.v_cmd = v_new
        if L.v_quantum > 0:
            self.v_sent = round(v_new / L.v_quantum) * L.v_quantum
        else:
            self.v_sent = v_new
        return self.v_sent, a_eff
