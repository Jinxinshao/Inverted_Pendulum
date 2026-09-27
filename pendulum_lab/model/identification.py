"""Identification of the pendulum from free-swing (hanging) experiments.

Two entry points:

* :func:`fit_decay_summary` - uses only what a stopwatch gives you:
  start amplitude, end amplitude, number of full cycles, total time.
* :func:`fit_free_decay` - uses a logged angle trace (``sensor-log`` CSV) and
  fits the natural frequency, viscous and Coulomb damping from all extrema.

Physics
-------
For the hanging pendulum (phi measured from the bottom, energy per unit J)

    phi_dd = -alpha sin(phi) - c phi_d - gamma sign(phi_d),   alpha = omega0^2
    E      = phi_d^2/2 + alpha (1 - cos phi)
    dE/dt  = -c phi_d^2 - gamma |phi_d|

* Period at amplitude A:  T(A) = T0 * 2 K(sin^2(A/2)) / pi   (K: complete
  elliptic integral, parameter m = k^2). At 90 deg the factor is 1.180, so
  dividing the stopwatch time by the cycle count overestimates T0 by ~10 %.
* Integrating dE/dt over one half cycle (extremum A_j to extremum A_{j+1}):

      alpha (cos A_j - cos A_{j+1}) = -c S(A) - gamma (A_j + A_{j+1})
      S(A) = int phi_d dphi = sqrt(2 alpha) * int_{-A}^{A} sqrt(cos phi - cos A) dphi

  which is LINEAR in (c, gamma) and exact for Coulomb friction; S is
  evaluated at the mean amplitude of the half cycle (error O(damping^2)).
  A linear-oscillator fit (A_{j+1} = r A_j - d) is biased at large amplitude
  because the restoring torque is alpha sin(phi), not alpha phi.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.special import ellipk

_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def period_factor(amplitude: float | np.ndarray) -> float | np.ndarray:
    """T(A)/T0 for a pendulum of amplitude ``A`` [rad]."""
    k2 = np.sin(np.asarray(amplitude) / 2.0) ** 2
    return 2.0 * ellipk(k2) / np.pi


def half_cycle_action(amplitude: float | np.ndarray, alpha: float) -> np.ndarray:
    """S(A) = integral of phi_d^2 dt over one half cycle of amplitude A.

    Uses phi = A sin(u) so the integrand is smooth, then 64-point Gauss-Legendre.
    Small-A limit: pi A^2 omega0 / 2.
    """
    a = np.atleast_1d(np.asarray(amplitude, float))[:, None]
    u = _GL_X[None, :] * (math.pi / 2)
    phi = a * np.sin(u)
    integrand = np.sqrt(np.clip(np.cos(phi) - np.cos(a), 0.0, None)) * a * np.cos(u)
    val = (math.pi / 2) * integrand @ _GL_W
    return math.sqrt(2.0 * alpha) * val


def next_amplitude(a: float, alpha: float, c: float, gamma: float) -> float:
    """Amplitude after one half cycle from the energy balance (0 if the rod sticks)."""

    def f(x):
        am = 0.5 * (a + x)
        return alpha * (math.cos(a) - math.cos(x)) + c * float(half_cycle_action(am, alpha)[0]) + gamma * (a + x)

    # f(a) = 2 c S + 2 gamma a > 0; f(0) < 0 if the swing continues
    if f(0.0) >= 0.0:
        return 0.0
    return brentq(f, 0.0, a, xtol=1e-12)


def decay_sequence(a0: float, n_half: int, alpha: float, c: float, gamma: float) -> np.ndarray:
    amps = [a0]
    for _ in range(n_half):
        amps.append(next_amplitude(amps[-1], alpha, c, gamma))
        if amps[-1] == 0.0:
            break
    return np.array(amps)


def sequence_time(amps: np.ndarray, omega0: float) -> float:
    """Time for the half cycles in ``amps`` (mean-amplitude period rule)."""
    mean_amp = 0.5 * (amps[:-1] + amps[1:])
    return float(np.sum(0.5 * (2 * math.pi / omega0) * period_factor(mean_amp)))


@dataclass
class DecayFit:
    omega0: float
    period_small: float
    viscous_c: float
    coulomb_gamma: float
    model: str
    naive_period: float
    detail: dict

    def summary(self) -> str:
        return (
            f"model={self.model}: omega0={self.omega0:.4f} rad/s, T0={self.period_small:.4f} s "
            f"(stopwatch mean period {self.naive_period:.4f} s), "
            f"c={self.viscous_c:.4f} 1/s, gamma={self.coulomb_gamma:.4f} rad/s^2"
        )


def fit_decay_summary(
    a0_deg: float = 90.0,
    a_end_deg: float = 50.0,
    n_cycles: int = 20,
    total_time: float = 24.85,
    model: str = "viscous",
) -> DecayFit:
    """Fit omega0 and ONE damping coefficient from a stopwatch summary.

    With only two amplitudes the viscous and Coulomb models cannot be told apart
    (see :func:`fit_decay_summary_both`); log the full swing and use
    :func:`fit_free_decay` to separate them.
    """
    if model not in ("viscous", "coulomb"):
        raise ValueError(model)
    a0, a1 = math.radians(a0_deg), math.radians(a_end_deg)
    n_half = 2 * n_cycles
    w0 = 2 * math.pi * n_cycles / total_time  # start from the naive estimate
    k = 0.0
    for _ in range(30):
        alpha = w0 * w0

        def end_err(kk):
            c, g = (kk, 0.0) if model == "viscous" else (0.0, kk)
            seq = decay_sequence(a0, n_half, alpha, c, g)
            return (seq[-1] if len(seq) == n_half + 1 else 0.0) - a1

        k = brentq(end_err, 0.0, 5.0, xtol=1e-12)
        c, g = (k, 0.0) if model == "viscous" else (0.0, k)
        seq = decay_sequence(a0, n_half, alpha, c, g)
        w_new = w0 * sequence_time(seq, w0) / total_time
        if abs(w_new - w0) < 1e-10:
            w0 = w_new
            break
        w0 = w_new
    c, g = (k, 0.0) if model == "viscous" else (0.0, k)
    seq = decay_sequence(a0, n_half, w0 * w0, c, g)
    return DecayFit(
        omega0=w0,
        period_small=2 * math.pi / w0,
        viscous_c=c,
        coulomb_gamma=g,
        model=model,
        naive_period=total_time / n_cycles,
        detail={
            "mean_period_factor": total_time / n_cycles / (2 * math.pi / w0),
            "a0_deg": a0_deg,
            "a_end_deg": a_end_deg,
            "amplitudes_deg": np.degrees(seq[::2]).round(2).tolist(),
        },
    )


def fit_decay_summary_both(**kw) -> tuple[DecayFit, DecayFit]:
    return fit_decay_summary(model="viscous", **kw), fit_decay_summary(model="coulomb", **kw)


def simulate_free_swing(
    omega0: float,
    c: float = 0.0,
    gamma: float = 0.0,
    a0: float = math.pi / 2,
    duration: float = 30.0,
    fs: float = 500.0,
    eps: float = 1e-3,
    d: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference ODE solution of the hanging pendulum (phi from the bottom).

    Coulomb friction is smoothed with tanh(phi_d/eps) to keep the ODE regular.
    """
    alpha = omega0 * omega0

    def f(_t, y):
        return [y[1], -alpha * math.sin(y[0]) - c * y[1] - gamma * math.tanh(y[1] / eps) - d * y[1] * abs(y[1])]

    t = np.arange(0.0, duration, 1.0 / fs)
    sol = solve_ivp(f, (0.0, duration), [a0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12, method="LSODA")
    return sol.t, sol.y[0]


def extrema(t: np.ndarray, phi: np.ndarray, min_amp: float = math.radians(3.0), min_sep: float = 0.2):
    """Return times and signed values of alternating extrema of ``phi``.

    ``phi`` may contain NaN (e.g. potentiometer dead zone); NaNs are replaced by
    0 (the bottom) before the peak search, which cannot create a false extremum
    larger than ``min_amp``. Each extremum is refined with a parabola.
    """
    t = np.asarray(t, float)
    y = np.nan_to_num(np.asarray(phi, float), nan=0.0)
    fs = 1.0 / np.median(np.diff(t))
    dist = max(1, int(min_sep * fs))
    idx = np.concatenate([find_peaks(y, height=min_amp, distance=dist)[0], find_peaks(-y, height=min_amp, distance=dist)[0]])
    idx.sort()
    tt, vv = [], []
    for i in idx:
        if 0 < i < len(y) - 1:
            y0, y1, y2 = y[i - 1], y[i], y[i + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            d = float(np.clip(d, -1.0, 1.0))
            tt.append(t[i] + d / fs)
            vv.append(y1 - 0.25 * (y0 - y2) * d)
        else:
            tt.append(t[i])
            vv.append(y[i])
    keep_t: list[float] = []
    keep_v: list[float] = []
    for ti, vi in zip(tt, vv):
        if keep_v and np.sign(vi) == np.sign(keep_v[-1]):
            if abs(vi) > abs(keep_v[-1]):
                keep_t[-1], keep_v[-1] = ti, vi
            continue
        keep_t.append(ti)
        keep_v.append(vi)
    return np.array(keep_t), np.array(keep_v)


def fit_free_decay_one_sided(t: np.ndarray, phi: np.ndarray, side: int, min_amp_deg: float = 5.0) -> DecayFit:
    """Like :func:`fit_free_decay` but using only the extrema on one side (sign ``side``).

    Needed on this rig: readings beyond the potentiometer dead zone wrap around
    the resistive track, whose full-turn count (4095*360/345 = 4273) differs from
    2*pi*k_cal (~4252) - extrema on the wrapped side carry a ~1.8 deg offset,
    while the per-half-cycle decrement is only ~0.4 deg. Same-side extrema are
    immune: periods are full periods, and the energy balance runs over a full
    cycle with the opposite-side amplitude approximated by the mean
    A_mid = (A_j + A_{j+2}) / 2 (error O(decrement^2)).
    """
    te, ve = extrema(t, phi, min_amp=math.radians(min_amp_deg))
    sel = np.sign(ve) == side
    te, amps = te[sel], np.abs(ve[sel])
    if len(te) < 4:
        raise ValueError("need at least 4 same-side extrema")
    full = np.diff(te)
    mean_amp = 0.5 * (amps[:-1] + amps[1:])
    t0_each = full / period_factor(mean_amp)
    T0 = float(np.median(t0_each))
    good = np.abs(t0_each / T0 - 1.0) < 0.1
    if good.sum() < 3:
        raise ValueError("fewer than 3 valid full cycles")
    T0 = float(np.median(t0_each[good]))
    w0 = 2 * math.pi / T0
    alpha = w0 * w0
    a0, a2 = amps[:-1][good], amps[1:][good]
    am = 0.5 * (a0 + a2)
    lhs = alpha * (np.cos(a0) - np.cos(a2))
    S = half_cycle_action(0.5 * (a0 + am), alpha) + half_cycle_action(0.5 * (am + a2), alpha)
    X = np.column_stack([S, a0 + 2 * am + a2])
    (c, gamma), *_ = np.linalg.lstsq(X, -lhs, rcond=None)
    if c < 0:
        c, gamma = 0.0, float(np.linalg.lstsq(X[:, 1:], -lhs, rcond=None)[0][0])
    if gamma < 0:
        gamma, c = 0.0, float(np.linalg.lstsq(X[:, :1], -lhs, rcond=None)[0][0])
    resid = X @ np.array([c, gamma]) + lhs
    return DecayFit(
        omega0=w0, period_small=T0, viscous_c=float(max(c, 0.0)), coulomb_gamma=float(max(gamma, 0.0)),
        model=f"one-sided({'+' if side > 0 else '-'}) viscous+coulomb", naive_period=float(np.mean(full)),
        detail={"n_full_cycles_used": int(good.sum()), "T0_spread_s": float(np.std(t0_each[good])),
                "energy_fit_rms": float(np.sqrt(np.mean(resid**2))),
                "first_amp_deg": math.degrees(amps[0]), "last_amp_deg": math.degrees(amps[-1])},
    )


def fit_free_decay(t: np.ndarray, phi: np.ndarray, min_amp_deg: float = 5.0) -> DecayFit:
    """Fit omega0, viscous c and Coulomb gamma from a logged free swing.

    ``phi`` is the angle from the BOTTOM in rad (NaN allowed for invalid samples).
    Needs at least ~6 half cycles with amplitude above ``min_amp_deg``; a swing
    decaying from ~90 deg to ~10 deg separates viscous from Coulomb well.
    """
    te, ve = extrema(t, phi, min_amp=math.radians(min_amp_deg))
    if len(te) < 4:
        raise ValueError("need at least 4 alternating extrema; log a longer / larger swing")
    amps = np.abs(ve)
    half = np.diff(te)
    mean_amp = 0.5 * (amps[:-1] + amps[1:])
    t0_each = 2.0 * half / period_factor(mean_amp)
    T0 = float(np.median(t0_each))
    # keep only true half cycles: a missing extremum (dead zone, small swing) makes
    # a "pair" span a full cycle - its implied T0 is ~2x off and it is rejected here
    good = np.abs(t0_each / T0 - 1.0) < 0.15
    if good.sum() < 3:
        raise ValueError("fewer than 3 valid half cycles")
    T0 = float(np.median(t0_each[good]))
    w0 = 2 * math.pi / T0
    alpha = w0 * w0

    # energy balance per half cycle, linear in (c, gamma)
    a_j, a_j1, m_a = amps[:-1][good], amps[1:][good], mean_amp[good]
    lhs = alpha * (np.cos(a_j) - np.cos(a_j1))  # = -(c S + gamma (A_j + A_j+1))
    S = half_cycle_action(m_a, alpha)
    X = np.column_stack([S, a_j + a_j1])
    (c, gamma), *_ = np.linalg.lstsq(X, -lhs, rcond=None)
    # physical constraint: both >= 0; refit the other one if one goes negative
    if c < 0:
        c, gamma = 0.0, float(np.linalg.lstsq(X[:, 1:], -lhs, rcond=None)[0][0])
    if gamma < 0:
        gamma, c = 0.0, float(np.linalg.lstsq(X[:, :1], -lhs, rcond=None)[0][0])
    resid = X @ np.array([c, gamma]) + lhs
    return DecayFit(
        omega0=w0,
        period_small=T0,
        viscous_c=float(max(c, 0.0)),
        coulomb_gamma=float(max(gamma, 0.0)),
        model="viscous+coulomb",
        naive_period=float(2 * np.mean(half)),
        detail={
            "n_extrema": int(len(te)),
            "n_half_cycles_used": int(good.sum()),
            "T0_spread_s": float(np.std(t0_each[good])),
            "energy_fit_rms": float(np.sqrt(np.mean(resid**2))),
            "first_amp_deg": math.degrees(amps[0]),
            "last_amp_deg": math.degrees(amps[-1]),
        },
    )


def omega0_from_period_at_amplitude(period: float, amplitude_deg: float) -> float:
    """Invert the large-amplitude period formula for one measured period."""
    return 2 * math.pi * float(period_factor(math.radians(amplitude_deg))) / period
