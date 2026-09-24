"""Physical parameters of the cart-pendulum.

Model convention (used everywhere in this package)
--------------------------------------------------
* ``x``      cart position [m], positive toward the motor's positive direction.
* ``theta``  pendulum angle [rad] measured from the UPRIGHT position,
             positive when the rod leans toward +x.
* The cart is driven kinematically by a closed-loop stepper, so the control
  input is the cart acceleration ``a`` [m/s^2].

With these conventions the pendulum obeys

    J*theta_dd = m*g*l*sin(theta) - m*l*cos(theta)*a - b*theta_d - Tc*sign(theta_d)

Dividing by J gives the form actually used in the code:

    theta_dd = alpha*sin(theta) - beta*cos(theta)*a - c*theta_d - gamma*sign(theta_d)

    alpha = m*g*l/J = omega0^2        (omega0: small-angle natural frequency when hanging)
    beta  = m*l/J   = alpha/g = 1/L_eq
    c     = b/J                       (viscous pivot damping)
    gamma = Tc/J                      (Coulomb pivot friction)

The key consequence: under kinematic actuation the whole pendulum model is fixed
by ``omega0`` (measurable with a stopwatch) plus the small damping terms.
Masses are only needed to report motor force/torque.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

G = 9.81

STEEL_DENSITY = 7850.0      # kg/m^3
ALUMINIUM_DENSITY = 2700.0  # kg/m^3


@dataclass
class PendulumGeometry:
    """Geometry of the student-built pendulum (all SI units).

    Defaults are the values reported for the lab rig: 8 mm x 1 mm steel tube,
    50 cm long, pivot clamped ~12 cm from the lower end, 3 cm aluminium head
    (OD 20 mm, bore 10 mm) at the upper end.
    """

    tube_od: float = 0.008
    tube_wall: float = 0.001
    tube_length: float = 0.50
    pivot_from_bottom: float = 0.12
    tube_density: float = STEEL_DENSITY
    head_length: float = 0.030
    head_od: float = 0.020
    head_id: float = 0.010
    head_density: float = ALUMINIUM_DENSITY

    def mass_properties(self) -> dict:
        """Return mass, COM distance above the pivot and inertia about the pivot."""
        r_o = self.tube_od / 2.0
        r_i = r_o - self.tube_wall
        lam = self.tube_density * math.pi * (r_o**2 - r_i**2)  # kg/m
        l_up = self.tube_length - self.pivot_from_bottom
        l_dn = self.pivot_from_bottom
        m_up = lam * l_up
        m_dn = lam * l_dn

        h_ro, h_ri = self.head_od / 2.0, self.head_id / 2.0
        m_head = self.head_density * math.pi * (h_ro**2 - h_ri**2) * self.head_length
        z_head = l_up - self.head_length / 2.0  # head flush with the top end

        m = m_up + m_dn + m_head
        moment = m_up * l_up / 2.0 - m_dn * l_dn / 2.0 + m_head * z_head
        l_c = moment / m

        j_up = m_up * l_up**2 / 3.0
        j_dn = m_dn * l_dn**2 / 3.0
        j_head_own = m_head * (3.0 * (h_ro**2 + h_ri**2) + self.head_length**2) / 12.0
        j_head = m_head * z_head**2 + j_head_own
        J = j_up + j_dn + j_head

        omega0 = math.sqrt(m * G * l_c / J)
        return {
            "mass_tube_upper": m_up,
            "mass_tube_lower": m_dn,
            "mass_head": m_head,
            "mass": m,
            "l_c": l_c,
            "J_pivot": J,
            "omega0": omega0,
            "period_small": 2.0 * math.pi / omega0,
            "L_eq": G / omega0**2,
        }


@dataclass
class PlantParams:
    """Parameters used by the simulator and by every controller design.

    ``omega0`` is the authoritative quantity. ``mass``/``l_c``/``J`` are kept
    for force/torque reporting and must be consistent with it
    (``omega0^2 = m g l_c / J``); :meth:`consistent` checks that.
    """

    omega0: float = 5.55            # rad/s, hanging small-amplitude natural frequency
    viscous_c: float = 0.028        # 1/s   (b/J)
    coulomb_gamma: float = 0.0      # rad/s^2 (Tc/J)
    mass: float = 0.1054            # kg, pendulum
    l_c: float = 0.1726             # m, COM above pivot
    J: float = 5.80e-3              # kg m^2, about pivot
    cart_mass: float = 0.120        # kg, 3D-printed cart + potentiometer + bracket
    g: float = G
    source: str = "geometry+free-swing summary"
    notes: list[str] = field(default_factory=list)

    @property
    def alpha(self) -> float:
        return self.omega0**2

    @property
    def beta(self) -> float:
        return self.omega0**2 / self.g

    @property
    def L_eq(self) -> float:
        return self.g / self.omega0**2

    @property
    def unstable_pole(self) -> float:
        """Open-loop unstable pole of the upright pendulum (undamped)."""
        c = self.viscous_c
        return -c / 2.0 + math.sqrt(c * c / 4.0 + self.alpha)

    def consistent(self, rel_tol: float = 0.05) -> bool:
        w = math.sqrt(self.mass * self.g * self.l_c / self.J)
        return abs(w - self.omega0) / self.omega0 < rel_tol

    def cart_force(self, a: float, theta: float, theta_d: float, theta_dd: float) -> float:
        """Horizontal force the belt must apply to cart+pendulum [N]."""
        m, l = self.mass, self.l_c
        return (self.cart_mass + m) * a + m * l * (theta_dd * math.cos(theta) - theta_d**2 * math.sin(theta))

    # ------------------------------------------------------------------ IO
    def to_dict(self) -> dict:
        d = asdict(self)
        d["derived"] = {
            "alpha": self.alpha,
            "beta": self.beta,
            "L_eq": self.L_eq,
            "unstable_pole": self.unstable_pole,
            "period_small": 2 * math.pi / self.omega0,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PlantParams":
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keys})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PlantParams":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def default_params() -> PlantParams:
    """Plant parameters for the lab rig, from geometry, cross-checked with the swing test.

    Geometry gives omega0 = 5.55 rad/s (T0 = 1.133 s). The measured summary
    (20 cycles from 90 deg decaying to 50 deg in 24.85 s) gives, after the
    large-amplitude period correction, T0 = 1.12-1.13 s; see
    :func:`pendulum_lab.model.identification.fit_decay_summary`.
    """
    mp = PendulumGeometry().mass_properties()
    return PlantParams(
        omega0=mp["omega0"],
        mass=mp["mass"],
        l_c=mp["l_c"],
        J=mp["J_pivot"],
    )
