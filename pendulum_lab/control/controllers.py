"""Controllers for the kinematically driven inverted pendulum.

Every controller returns the desired cart ACCELERATION a_cmd [m/s^2]; the
actuator interface integrates it into the velocity command sent to the PD42S1.
Signs: a positive a_cmd moves the cart toward +x; theta > 0 means the rod
leans toward +x. To catch a rod leaning to +x the cart must accelerate to +x,
so all angle gains are POSITIVE in u = sum(g_i * s_i).

Linear controllers expose a :class:`LinearLaw` so the same object can be
simulated, run on hardware and analysed (closed-loop poles, delay margin).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..model.dynamics import energy_upright, linear_model, wrap_angle
from ..model.params import PlantParams
from .discrete import ackermann, augment, discretize_with_delay, dlqr


@dataclass
class ControlInput:
    t: float
    x: float
    v: float
    theta: float
    omega: float
    v_cmd: float = 0.0
    u_hist: deque = field(default_factory=lambda: deque(maxlen=32))  # u_{k-1}, u_{k-2}, ...
    x_ref: float = 0.0

    def phys(self) -> np.ndarray:
        return np.array([self.x - self.x_ref, self.v, self.theta, self.omega])


@dataclass
class LinearLaw:
    """u = k_state.[x-xr, v, th, om] + k_vcmd*v_cmd + k_hist.[u_{k-1}..] + Cq.q
    q_{k+1} = Aq q + Bq.[x-xr, v, th, om]"""

    k_state: np.ndarray
    k_vcmd: float = 0.0
    k_hist: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Aq: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Bq: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    Cq: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_q(self) -> int:
        return self.Aq.shape[0]


class Controller:
    name = "base"
    label = "base"

    def reset(self) -> None:
        pass

    def update(self, ci: ControlInput) -> float:
        raise NotImplementedError

    def law(self) -> LinearLaw | None:
        """Linear law for analysis (None for nonlinear controllers)."""
        return None

    def describe(self) -> dict:
        return {"name": self.name}


class LinearController(Controller):
    """Applies a :class:`LinearLaw` (with its internal states)."""

    def __init__(self, law: LinearLaw):
        self._law = law
        self.q = np.zeros(law.n_q)

    def law(self) -> LinearLaw:
        return self._law

    def reset(self) -> None:
        self.q = np.zeros(self._law.n_q)

    def update(self, ci: ControlInput) -> float:
        L = self._law
        s = ci.phys()
        u = float(L.k_state @ s) + L.k_vcmd * ci.v_cmd
        if len(L.k_hist):
            h = np.zeros(len(L.k_hist))
            for i, val in enumerate(list(ci.u_hist)[: len(L.k_hist)]):
                h[i] = val
            u += float(L.k_hist @ h)
        if L.n_q:
            u += float(L.Cq @ self.q)
            self.q = L.Aq @ self.q + L.Bq @ s
        return u


# --------------------------------------------------------------------------- PD
class AnglePD(LinearController):
    """a = Kp*theta + Kd*omega (+ optional weak cart centring kx, kv).

    Closed loop (linear, ideal actuator):
        theta_dd = (alpha - beta Kp) theta - (c + beta Kd) theta_d
    Stable iff Kp > alpha/beta = g  and  Kd > -c/beta.
    Without cart terms the cart position is NOT controlled: any angle offset
    delta makes the cart accelerate at g*delta and run into the rail.
    """

    name = "pd"
    label = "PD (angle only)"

    def __init__(self, kp: float, kd: float, kx: float = 0.0, kv: float = 0.0):
        self.kp, self.kd, self.kx, self.kv = kp, kd, kx, kv
        super().__init__(LinearLaw(k_state=np.array([kx, kv, kp, kd])))

    @classmethod
    def from_poles(cls, p: PlantParams, wn: float = 8.0, zeta: float = 1.0) -> "AnglePD":
        kp = (p.alpha + wn * wn) / p.beta
        kd = (2 * zeta * wn - p.viscous_c) / p.beta
        return cls(kp, kd)

    def describe(self) -> dict:
        return {"name": self.name, "Kp [m/s^2/rad]": self.kp, "Kd [m/s^2/(rad/s)]": self.kd, "kx": self.kx, "kv": self.kv}


# ------------------------------------------------------------------ cascade PID
class CascadePID(Controller):
    """Inner PID on the angle error, outer PD on the cart position.

        theta_ref = clamp(-(kx (x - x_ref) + kv v), +-theta_ref_max)
        e         = theta - theta_ref
        a         = Kp e + Ki int(e) + Kd omega

    Quasi-static outer loop: a balanced rod at angle theta accelerates the cart
    at a = g theta, so with a fast inner loop x_dd = -g (kx x + kv v).
    Choose g kx = wo^2, g kv = 2 zeta_o wo with wo several times slower than
    the inner loop. The cart first moves AWAY from the target (non-minimum
    phase) because the rod has to be tilted toward it first.

    Exact (ideal actuator) characteristic polynomial of u = k1 x + k2 v + k3 th + k4 om:
        s^4 + (beta k4 - k2) s^3 + (beta k3 - alpha - k1) s^2 + alpha k2 s + alpha k1
    so the cascade needs  Kp kv < beta Kd  (outer velocity gain below the inner
    damping): the time-scale separation is a hard stability condition, not a nicety.
    """

    name = "pid"
    label = "Cascade PID"

    def __init__(self, kp, ki, kd, kx, kv, theta_ref_max=math.radians(6.0), i_limit=math.radians(3.0), Ts=0.005):
        self.kp, self.ki, self.kd, self.kx, self.kv = kp, ki, kd, kx, kv
        self.theta_ref_max = theta_ref_max
        self.i_limit = i_limit  # clamp on the integral of e [rad*s] -> anti-windup
        self.Ts = Ts
        self.z = 0.0

    @classmethod
    def design(cls, p: PlantParams, Ts: float, wn: float = 8.0, zeta: float = 1.0, wo: float = 0.8, zeta_o: float = 1.0, ki: float = 0.0):
        kp = (p.alpha + wn * wn) / p.beta
        kd = (2 * zeta * wn - p.viscous_c) / p.beta
        kx = wo * wo / p.g
        kv = 2 * zeta_o * wo / p.g
        return cls(kp, ki, kd, kx, kv, Ts=Ts)

    def reset(self):
        self.z = 0.0

    def theta_ref(self, ci: ControlInput) -> float:
        r = -(self.kx * (ci.x - ci.x_ref) + self.kv * ci.v)
        return max(-self.theta_ref_max, min(self.theta_ref_max, r))

    def update(self, ci: ControlInput) -> float:
        e = ci.theta - self.theta_ref(ci)
        u = self.kp * e + self.ki * self.z + self.kd * ci.omega
        self.z = max(-self.i_limit, min(self.i_limit, self.z + self.Ts * e))
        return u

    def law(self) -> LinearLaw:
        # u = kp(th + kx x + kv v) + kd om + ki z ; z+ = z + Ts(th + kx x + kv v)
        row = np.array([self.kp * self.kx, self.kp * self.kv, self.kp, self.kd])
        e_row = np.array([self.kx, self.kv, 1.0, 0.0])
        if self.ki == 0.0:
            return LinearLaw(k_state=row)
        return LinearLaw(
            k_state=row,
            Aq=np.array([[1.0]]),
            Bq=(self.Ts * e_row)[None, :],
            Cq=np.array([self.ki]),
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "Kp": self.kp,
            "Ki": self.ki,
            "Kd": self.kd,
            "kx [rad/m]": self.kx,
            "kv [rad/(m/s)]": self.kv,
            "theta_ref_max_deg": math.degrees(self.theta_ref_max),
        }


# -------------------------------------------------------------- design models
def design_model(p: PlantParams, Ts: float, delay: float = 0.0, actuator_tau: float = 0.0, n_hist: int | None = None):
    """Augmented discrete model used by the model-based designs.

    State order: [x, v, theta, omega, (v_cmd if tau>0), u_{k-1}, ...]
    """
    A, B, names = linear_model(p, actuator_tau)
    dm = discretize_with_delay(A, B, Ts, delay)
    F, G = augment(dm, n_hist)
    n_phys = A.shape[0]
    names = list(names) + [f"u[k-{j}]" for j in range(1, F.shape[0] - n_phys + 1)]
    return F, G, n_phys, names


def _law_from_K(K: np.ndarray, n_phys: int, integrator: bool = False, Ts: float = 0.0) -> LinearLaw:
    """Convert u = -K xi into a LinearLaw (xi = [phys4, (v_cmd), hist..., (z)])."""
    g = -np.asarray(K).ravel()
    if integrator:
        g_z, g = g[-1], g[:-1]
    k_state = g[:4]
    k_vcmd = float(g[4]) if n_phys == 5 else 0.0
    k_hist = g[n_phys:]
    if not integrator:
        return LinearLaw(k_state=k_state, k_vcmd=k_vcmd, k_hist=k_hist)
    return LinearLaw(
        k_state=k_state,
        k_vcmd=k_vcmd,
        k_hist=k_hist,
        Aq=np.array([[1.0]]),
        Bq=np.array([[Ts, 0.0, 0.0, 0.0]]),
        Cq=np.array([g_z]),
    )


# -------------------------------------------------------------------------- LQR
class LQR(LinearController):
    """Discrete LQR on the (optionally delay- and lag-augmented) model.

    J = sum( q_x x^2 + q_v v^2 + q_th th^2 + q_om om^2 + r a^2 )  (+ q_i z^2 with LQI)
    Bryson's rule: q_i = 1/(max acceptable deviation)^2, r = 1/a_max^2.
    """

    name = "lqr"
    label = "LQR"

    def __init__(
        self,
        p: PlantParams,
        Ts: float,
        q=(20.0, 5.0, 200.0, 5.0),
        r: float = 0.2,
        design_delay: float = 0.0,
        actuator_tau: float = 0.0,
        q_integral: float = 0.0,
    ):
        self.p, self.Ts = p, Ts
        self.q_diag, self.r = tuple(q), r
        self.design_delay, self.actuator_tau, self.q_integral = design_delay, actuator_tau, q_integral
        F, G, n_phys, names = design_model(p, Ts, design_delay, actuator_tau)
        Q = np.zeros((F.shape[0], F.shape[0]))
        Q[:4, :4] = np.diag(q)
        if q_integral > 0:
            N = F.shape[0]
            Fi = np.zeros((N + 1, N + 1))
            Fi[:N, :N] = F
            Fi[N, 0] = Ts
            Fi[N, N] = 1.0
            Gi = np.vstack([G, [[0.0]]])
            Qi = np.zeros((N + 1, N + 1))
            Qi[:N, :N] = Q
            Qi[N, N] = q_integral
            K, _ = dlqr(Fi, Gi, Qi, np.array([[r]]))
            names = names + ["int(x)"]
        else:
            K, _ = dlqr(F, G, Q, np.array([[r]]))
        self.K = K
        self.state_names = names
        super().__init__(_law_from_K(K, n_phys, integrator=q_integral > 0, Ts=Ts))

    def describe(self) -> dict:
        d = {"name": self.name, "Q_diag": self.q_diag, "R": self.r, "design_delay_s": self.design_delay, "actuator_tau_s": self.actuator_tau}
        d["K (u=-K xi)"] = {n: float(k) for n, k in zip(self.state_names, self.K.ravel())}
        return d


# ------------------------------------------------------------ pole placement
class PolePlacement(LinearController):
    """State feedback placing the closed-loop poles of the 4 physical modes.

    Continuous poles s_i are mapped to z_i = exp(s_i Ts). Extra states of the
    augmented model (actuator-lag state, delayed inputs) get z = 0 (deadbeat)
    or the actuator pole exp(-Ts/tau).
    """

    name = "place"
    label = "Pole placement"

    def __init__(self, p: PlantParams, Ts: float, poles=(-8 + 6j, -8 - 6j, -3.0, -2.0), design_delay: float = 0.0, actuator_tau: float = 0.0):
        self.p, self.Ts, self.poles = p, Ts, tuple(poles)
        self.design_delay, self.actuator_tau = design_delay, actuator_tau
        F, G, n_phys, names = design_model(p, Ts, design_delay, actuator_tau)
        zs = [np.exp(complex(s) * Ts) for s in poles]
        if n_phys == 5:
            zs.append(math.exp(-Ts / actuator_tau))
        zs += [0.0] * (F.shape[0] - len(zs))
        self.K = ackermann(F, G, np.array(zs))
        self.state_names = names
        super().__init__(_law_from_K(self.K, n_phys))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "poles_continuous": [str(complex(s)) for s in self.poles],
            "K (u=-K xi)": {n: float(k) for n, k in zip(self.state_names, self.K.ravel())},
        }


# ------------------------------------------------------------- swing-up (sim)
class EnergySwingUp(Controller):
    """Energy-shaping swing-up with a linear catch controller.

    Energy per unit inertia (zero at upright rest):
        e = omega^2/2 + alpha (cos theta - 1),     de/dt = -beta a omega cos(theta) - (friction)
    Choosing a = k e omega cos(theta) (k > 0) gives de/dt = -beta k e (omega cos theta)^2,
    which drives e -> 0 (the homoclinic orbit through the upright position).
    A weak PD keeps the cart near the middle of the rail. Inside the catch
    region the linear controller takes over.

    Hardware caveat: the potentiometer's 15 deg electrical dead zone lies right
    next to the hanging position on this rig; swing-up needs the angle
    estimator to bridge it. Use it in simulation first.
    """

    name = "swingup"
    label = "Energy swing-up + LQR catch"

    def __init__(self, p: PlantParams, catch: Controller, k_energy: float = 0.06, a_max: float = 3.0, kx: float = 15.0, kv: float = 6.0,
                 catch_angle: float = math.radians(25.0), release_angle: float = math.radians(40.0)):
        self.p, self.catch = p, catch
        self.k_energy, self.a_max, self.kx, self.kv = k_energy, a_max, kx, kv
        self.catch_angle, self.release_angle = catch_angle, release_angle
        self.mode = "swing"

    def reset(self):
        self.mode = "swing"
        self.catch.reset()

    def update(self, ci: ControlInput) -> float:
        th = wrap_angle(ci.theta)
        if self.mode == "swing" and abs(th) < self.catch_angle:
            self.mode = "catch"
            self.catch.reset()
        elif self.mode == "catch" and abs(th) > self.release_angle:
            self.mode = "swing"
        if self.mode == "catch":
            c2 = ControlInput(ci.t, ci.x, ci.v, th, ci.omega, ci.v_cmd, ci.u_hist, ci.x_ref)
            return self.catch.update(c2)
        e = energy_upright(self.p, th, ci.omega)
        a = self.k_energy * e * ci.omega * math.cos(th)
        # kick-start from rest at the bottom
        if abs(ci.omega) < 0.05 and abs(abs(th) - math.pi) < 0.05:
            a = self.a_max
        a = max(-self.a_max, min(self.a_max, a))
        return a - self.kx * (ci.x - ci.x_ref) - self.kv * ci.v

    def describe(self) -> dict:
        return {"name": self.name, "k_energy": self.k_energy, "a_max": self.a_max, "catch": self.catch.describe()}


class NullController(Controller):
    name = "none"
    label = "No control (open loop)"

    def update(self, ci: ControlInput) -> float:
        return 0.0

    def law(self) -> LinearLaw:
        return LinearLaw(k_state=np.zeros(4))
