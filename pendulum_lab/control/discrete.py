"""Discrete-time tools: exact ZOH discretisation with (fractional) input delay,
delay-augmented models, DLQR, Ackermann pole placement, steady-state Kalman gain
and closed-loop eigen-analysis.

Delay model
-----------
A computer-controlled loop applies u_k with a total latency tau (sensor
transport + computation + command transmission + driver reaction). Write
tau = d*Ts + f*Ts with integer d >= 0 and 0 <= f < 1. Over one sample
interval [kTs, (k+1)Ts) the plant sees u_{k-d-1} for f*Ts, then u_{k-d}.
Exact ZOH integration gives

    s_{k+1} = Phi s_k + sum_j Gamma_j u_{k-j}

with Phi = e^{A Ts},
     Gamma_d     = int_0^{(1-f)Ts} e^{A eta} d eta B
     Gamma_{d+1} = e^{A (1-f) Ts} int_0^{f Ts} e^{A eta} d eta B
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_discrete_are


def _zoh_integral(A: np.ndarray, B: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (e^{AT}, int_0^T e^{A eta} d eta B) via the block-matrix exponential."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * T)
    return E[:n, :n], E[:n, n:]


def c2d(A: np.ndarray, B: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    return _zoh_integral(A, B, Ts)


@dataclass
class DelayedModel:
    """s_{k+1} = Phi s_k + sum_j Gammas[j] u_{k-j}  (j = 0..len(Gammas)-1)."""

    Phi: np.ndarray
    Gammas: list[np.ndarray]
    Ts: float
    delay: float

    @property
    def n(self) -> int:
        return self.Phi.shape[0]

    @property
    def n_hist(self) -> int:
        """Number of past inputs (u_{k-1}...) that influence the next state."""
        return len(self.Gammas) - 1


def discretize_with_delay(A: np.ndarray, B: np.ndarray, Ts: float, delay: float = 0.0) -> DelayedModel:
    if delay < 0:
        raise ValueError("delay must be >= 0")
    d = int(np.floor(delay / Ts + 1e-9))
    f = delay / Ts - d
    if f < 1e-9:
        f = 0.0
    Phi, G_full = _zoh_integral(A, B, Ts)
    n = A.shape[0]
    gammas = [np.zeros((n, B.shape[1])) for _ in range(d + (2 if f > 0 else 1))]
    if f == 0.0:
        gammas[d] = G_full
    else:
        E1, G1 = _zoh_integral(A, B, (1.0 - f) * Ts)
        _, G2 = _zoh_integral(A, B, f * Ts)
        gammas[d] = G1
        gammas[d + 1] = E1 @ G2
    return DelayedModel(Phi, gammas, Ts, delay)


def augment(model: DelayedModel, n_hist: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Augmented model with state [s, u_{k-1}, ..., u_{k-h}] and input u_k.

    ``n_hist`` may exceed the model's own history length (extra zero columns);
    this is needed when a controller uses more past inputs than the plant.
    """
    h = model.n_hist if n_hist is None else max(n_hist, model.n_hist)
    n = model.n
    N = n + h
    F = np.zeros((N, N))
    G = np.zeros((N, 1))
    F[:n, :n] = model.Phi
    G[:n, :] = model.Gammas[0]
    for j in range(1, len(model.Gammas)):
        F[:n, n + j - 1 : n + j] = model.Gammas[j]
    if h > 0:
        G[n, 0] = 1.0
        for j in range(1, h):
            F[n + j, n + j - 1] = 1.0
    return F, G


def dlqr(F: np.ndarray, G: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discrete LQR: u = -K xi minimising sum xi'Q xi + u'R u. Returns (K, P)."""
    P = solve_discrete_are(F, G, Q, R)
    K = np.linalg.solve(R + G.T @ P @ G, G.T @ P @ F)
    return K, P


def dlqe(F: np.ndarray, H: np.ndarray, Qw: np.ndarray, Rv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Steady-state Kalman gain (predictor form P, filter gain L).

    x_{k+1} = F x_k + w,  y_k = H x_k + v,  cov(w)=Qw, cov(v)=Rv.
    Returns (L, P) with the measurement update x+ = x- + L (y - H x-).
    """
    P = solve_discrete_are(F.T, H.T, Qw, Rv)
    S = H @ P @ H.T + Rv
    L = P @ H.T @ np.linalg.inv(S)
    return L, P


def controllability_matrix(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    cols = [G]
    for _ in range(n - 1):
        cols.append(F @ cols[-1])
    return np.hstack(cols)


def ackermann(F: np.ndarray, G: np.ndarray, poles: np.ndarray) -> np.ndarray:
    """Single-input pole placement (u = -K x). Poles may repeat (e.g. delay states at z=0)."""
    n = F.shape[0]
    C = controllability_matrix(F, G)
    if np.linalg.matrix_rank(C) < n:
        raise ValueError("model not controllable")
    coeffs = np.real(np.poly(poles))
    phi = np.zeros_like(F)
    Fp = np.eye(n)
    for c in coeffs[::-1]:
        phi = phi + c * Fp
        Fp = Fp @ F
    e = np.zeros((1, n))
    e[0, -1] = 1.0
    return e @ np.linalg.solve(C, phi)


def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))
