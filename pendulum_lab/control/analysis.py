"""Closed-loop linear analysis of any :class:`LinearLaw` against an evaluation
plant that may differ from the design model (sampling period, delay, lag).

Assumes the controller sees the true physical state (separation principle);
the estimator's own dynamics are examined in simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..model.dynamics import linear_model
from ..model.params import PlantParams
from .controllers import LinearLaw
from .discrete import augment, discretize_with_delay


def closed_loop_matrix(p: PlantParams, law: LinearLaw, Ts: float, delay: float = 0.0, actuator_tau: float = 0.0) -> np.ndarray:
    A, B, _ = linear_model(p, actuator_tau)
    n_phys = A.shape[0]
    dm = discretize_with_delay(A, B, Ts, delay)
    h = max(dm.n_hist, len(law.k_hist))
    F, G = augment(dm, h)
    N = F.shape[0]
    Kxi = np.zeros(N)
    Kxi[:4] += law.k_state
    if n_phys == 5:
        Kxi[4] += law.k_vcmd
    else:  # ideal actuator: v_cmd == v
        Kxi[1] += law.k_vcmd
    Kxi[n_phys : n_phys + len(law.k_hist)] += law.k_hist
    nq = law.n_q
    M = np.zeros((N + nq, N + nq))
    M[:N, :N] = F + G @ Kxi[None, :]
    if nq:
        M[:N, N:] = G @ law.Cq[None, :]
        M[N:, :4] = law.Bq
        M[N:, N:] = law.Aq
    return M


@dataclass
class LoopReport:
    stable: bool
    spectral_radius: float
    z_poles: np.ndarray
    s_poles: np.ndarray
    delay_margin: float | None

    def lines(self) -> list[str]:
        out = [f"stable: {self.stable}   spectral radius |z|max = {self.spectral_radius:.4f}"]
        dm = "n/a" if self.delay_margin is None else f"{self.delay_margin * 1e3:.1f} ms"
        out.append(f"extra-delay margin (total latency tolerated before instability): {dm}")
        out.append("dominant closed-loop poles (continuous equivalent s = ln z / Ts):")
        order = np.argsort(-np.abs(self.z_poles))
        for i in order[:8]:
            z, s = self.z_poles[i], self.s_poles[i]
            wn = abs(s)
            zeta = -s.real / wn if wn > 1e-9 else float("nan")
            out.append(f"   z = {z.real:+.4f}{z.imag:+.4f}j   s = {s.real:+8.3f}{s.imag:+8.3f}j   wn={wn:7.3f}  zeta={zeta:+.3f}")
        return out


def analyse(p: PlantParams, law: LinearLaw, Ts: float, delay: float = 0.0, actuator_tau: float = 0.0, margin_search: bool = True) -> LoopReport:
    M = closed_loop_matrix(p, law, Ts, delay, actuator_tau)
    z = np.linalg.eigvals(M)
    rho = float(np.max(np.abs(z)))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.log(z.astype(complex)) / Ts
    dm = delay_margin(p, law, Ts, actuator_tau) if margin_search else None
    return LoopReport(rho < 1.0, rho, z, s, dm)


def delay_margin(p: PlantParams, law: LinearLaw, Ts: float, actuator_tau: float = 0.0, max_delay: float = 0.3, step: float | None = None) -> float | None:
    """Largest total latency [s] for which the loop is still stable (fractional delays included)."""
    step = step or Ts / 4
    if np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(p, law, Ts, 0.0, actuator_tau)))) >= 1.0:
        return None
    lo, d = 0.0, step
    while d <= max_delay:
        rho = np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(p, law, Ts, d, actuator_tau))))
        if rho >= 1.0:
            # refine by bisection between lo and d
            hi = d
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                if np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(p, law, Ts, mid, actuator_tau)))) < 1.0:
                    lo = mid
                else:
                    hi = mid
            return lo
        lo = d
        d += step
    return max_delay


def continuous_open_loop_poles(p: PlantParams) -> np.ndarray:
    A, _, _ = linear_model(p)
    return np.linalg.eigvals(A)


def pd_stability_region(p: PlantParams) -> dict:
    """Closed-form continuous-time conditions for the angle PD (teaching aid)."""
    return {
        "Kp_min [m/s^2/rad]": p.alpha / p.beta,
        "Kd_min": -p.viscous_c / p.beta,
        "note": "Kp > g: the cart must out-accelerate gravity's toppling tendency (a = g*theta balances).",
    }


def rule_of_thumb_delay(p: PlantParams) -> float:
    """Latency above which an unstable pole p_u is practically uncontrollable (p_u*tau ~ 0.5)."""
    return 0.5 / p.unstable_pole


def fmt_deg(rad: float) -> str:
    return f"{math.degrees(rad):.2f} deg"
