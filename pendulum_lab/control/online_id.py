"""Parameter identification methods that work on streaming data.

1. :class:`SVF` - state-variable filter: a 2nd-order low-pass
       F(s) = wf^2 / (s^2 + 2 zeta wf s + wf^2)
   that delivers the filtered signal AND its 1st/2nd derivatives without
   numerical differentiation. Because F is linear and time-invariant, applying
   the same F to both sides of  y = alpha * phi  keeps the equation exact:
       y_f = alpha * phi_f      (the "filter both sides" trick)
   while removing the ADC quantisation noise that raw differencing would
   amplify by 1/Ts^2.

2. :func:`ls_free_swing` - offline least squares on the full hanging-pendulum
   ODE (omega0, viscous c, Coulomb gamma in one linear regression).

3. :class:`OnlineOmegaRLS` - closed-loop, real-time estimation of omega0
   while the rod is being balanced. Physics makes it a SCALAR problem:
       theta_dd + c theta_d = alpha * (sin theta - cos theta * a / g),  beta = alpha / g
   Generic closed-loop identification of (alpha, beta) is ill-conditioned
   because the feedback makes a ~ K theta (collinear regressors); the
   constraint beta = alpha/g (which holds for ANY rigid pendulum under
   kinematic cart drive) removes that problem.
   Closed-loop pitfalls measured in the twin (docs/03 section 8):
     * without external excitation the loop sits near the equilibrium and the
       estimate is dominated by noise fed back through the controller: +5..8 %
       bias with a large variance (the convergence test then blocks updates);
     * unmeasured disturbances (hand taps) are correlated with the feedback
       input -> -10..-20 % bias with a deceptively SMALL variance; residual
       outlier rejection detects them (>2 % rejected samples blocks updates);
     * with a KNOWN excitation (x_ref square wave +-3 cm) the estimate is within
       ~1 % of the truth for omega0 = 5.0 ... 6.0 rad/s.

4. :class:`AdaptiveUpdater` - certainty-equivalence self-tuning: when the RLS
   estimate has converged and differs from the design model, redesign the
   controller with the new omega0 (bounded and rate-limited).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm

from ..model.params import PlantParams


class SVF:
    """State-variable filter F(s) = wf^2/(s^2 + 2 zeta wf s + wf^2); returns (f, f_d, f_dd).

    Two exact discretisations, chosen by what the INPUT really is between samples:

    ``hold="zoh"`` (default) - the input is piecewise constant, e.g. a cart
    acceleration command held by the driver. With a ZOH input the continuous
    f_dd jumps by wf^2 * du at every sample (for wf = 30 rad/s a 17 % error),
    so the outputs are paired over the interval [k-1, k]:
        f_dd = (f_d[k] - f_d[k-1]) / Ts    (exact mean of f_dd over the interval)
        f, f_d = trapezoidal means over the same interval.

    ``hold="foh"`` - the input is a SMOOTH signal that was sampled (an angle).
    Linear interpolation between samples; the states at the sample instants
    are the exact response to the interpolant and f_dd = wf^2 (u - f) -
    2 zeta wf f_d is continuous. One more correction is needed: f_dd = s^2 F
    has the direct feed-through wf^2 u, so the interpolation error of u (mean
    -Ts^2 u_dd / 12 per interval, seen by f but not by the sample u_k) is
    amplified by wf^2 and sampled synchronously. The result is a constant
    relative error of f_dd of  -(wf Ts)^2 / 12  for every low-frequency signal
    (measured on noiseless pendulum data at Ts = 5 ms: alpha 0.08 / 0.33 /
    0.74 % low for wf = 20 / 40 / 60 rad/s, formula 0.08 / 0.33 / 0.75 %; the
    ZOH pairing has the same error). f_dd is divided by (1 - (wf Ts)^2 / 12).

    Both sides of a regression must be passed through the same SVF.
    """

    def __init__(self, Ts: float, wf: float = 20.0, zeta: float = 0.8, hold: str = "zoh"):
        if hold not in ("zoh", "foh"):
            raise ValueError("hold must be 'zoh' or 'foh'")
        A = np.array([[0.0, 1.0], [-wf * wf, -2 * zeta * wf]])
        B = np.array([[0.0], [wf * wf]])
        if hold == "zoh":
            M = np.zeros((3, 3))
            M[:2, :2], M[:2, 2:] = A, B
            E = expm(M * Ts)
            self.Ad, self.Bd = E[:2, :2], E[:2, 2]
            self.B1 = np.zeros(2)
        else:  # augmented state [x, u, du]: u' = du / Ts
            M = np.zeros((4, 4))
            M[:2, :2], M[:2, 2:3] = A, B
            M[2, 3] = 1.0 / Ts
            E = expm(M * Ts)
            self.Ad, self.Bd, self.B1 = E[:2, :2], E[:2, 2], E[:2, 3]
        self.A, self.Ts, self.hold = A, Ts, hold
        self.dd_gain = 1.0 / (1.0 - (wf * Ts) ** 2 / 12.0) if hold == "foh" else 1.0
        self.wf, self.zeta, self.wf2 = wf, zeta, wf * wf
        self.x = np.zeros(2)
        self.u_prev: float | None = None

    def reset(self, value: float = 0.0, slope: float = 0.0) -> None:
        self.x = np.array([value, slope])
        self.u_prev = value

    def step(self, u: float) -> tuple[float, float, float]:
        x_old = self.x
        if self.hold == "foh":
            up = u if self.u_prev is None else self.u_prev
            self.x = self.Ad @ self.x + self.Bd * up + self.B1 * (u - up)
            self.u_prev = u
            f, fd = self.x
            return float(f), float(fd), float(self.dd_gain * (self.wf2 * (u - f) - 2 * self.zeta * self.wf * fd))
        self.x = self.Ad @ self.x + self.Bd * u
        f = 0.5 * (x_old[0] + self.x[0])
        fd = 0.5 * (x_old[1] + self.x[1])
        fdd = (self.x[1] - x_old[1]) / self.Ts
        return f, fd, fdd


def svf_filter(signal: np.ndarray, Ts: float, wf: float = 20.0, zeta: float = 0.8, hold: str = "zoh") -> np.ndarray:
    """Filter a whole array; returns an (N, 3) array of (f, f_d, f_dd)."""
    s = SVF(Ts, wf, zeta, hold)
    s.reset(float(signal[0]))
    return np.array([s.step(float(u)) for u in signal])


@dataclass
class LSFit:
    omega0: float
    viscous_c: float
    coulomb_gamma: float
    rms_residual: float
    n: int
    quad_d: float = 0.0

    def summary(self) -> str:
        return (f"LS/SVF: omega0={self.omega0:.4f} rad/s, c={self.viscous_c:.4f} 1/s, "
                f"gamma={self.coulomb_gamma:.4f} rad/s^2, d={self.quad_d:.5f} 1/rad "
                f"(n={self.n}, rms residual {self.rms_residual:.3f})")


def _svf_from(signal: np.ndarray, Ts: float, wf: float, f0: float, fd0: float, hold: str = "foh") -> np.ndarray:
    """Filter a segment starting from state (f0, fd0) AT sample 0 (FOH) or before it (ZOH)."""
    s = SVF(Ts, wf, hold=hold)
    s.x = np.array([f0, fd0])
    if hold == "zoh":
        return np.array([s.step(float(u)) for u in signal])
    s.u_prev = float(signal[0])
    first = (f0, fd0, s.dd_gain * (s.wf2 * (float(signal[0]) - f0) - 2 * s.zeta * wf * fd0))
    return np.array([first] + [s.step(float(u)) for u in signal[1:]])


def _valid_runs(valid: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(valid) - start >= min_len:
        runs.append((start, len(valid)))
    return runs


def ls_free_swing(t: np.ndarray, phi: np.ndarray, wf: float = 60.0, min_rate: float = 0.05,
                  t_start: float | None = None, settle: float = 0.1, drag: bool = False) -> LSFit:
    """Least squares on  phi_dd = -alpha sin(phi) - c phi_d - gamma sign(phi_d)  (phi from the bottom).

    ``drag=True`` adds the air-drag term  - d phi_d |phi_d|  (regressor built from
    the filtered rate, i.e. F[phi_d |phi_d|] ~ f_d |f_d| - adequate for wf >> omega0).

    Runs the SVF separately over every contiguous VALID segment (the
    potentiometer dead zone splits the record every half cycle), discards the
    first ``settle`` seconds of each segment (filter start-up), and regresses
    on all remaining samples. sign(phi_d) uses the filtered rate.
    """
    t = np.asarray(t, float)
    phi = np.asarray(phi, float)
    Ts = float(np.median(np.diff(t)))
    valid = ~np.isnan(phi)
    if t_start is not None:
        valid &= t >= t_start + 0.3
    skip = int(round(settle / Ts))
    Xs, Ys = [], []
    for a, b in _valid_runs(valid, skip + 10):
        seg = phi[a:b]
        # start every filter at the signal's local value AND slope (quadratic fit of the
        # first samples); a filter started "at rest" on a moving signal adds a transient
        # that is not the same for all regressors and biases the damping terms badly
        n0 = min(len(seg), 15)
        tt = np.arange(n0) * Ts
        c2 = np.polyfit(tt, seg[:n0], 2)
        p0, pd0, pdd0 = c2[2], c2[1], 2 * c2[0]
        # steady-state tracking solution of F(s) for a locally quadratic input:
        #   f = u - (2 zeta/wf) u_d + ((4 zeta^2 - 1)/wf^2) u_dd,   f_d = u_d - (2 zeta/wf) u_dd
        # i.e. the state the filter WOULD have if it had been running -> no start-up transient
        z = 0.8
        k1, k2 = 2 * z / wf, (4 * z * z - 1) / wf**2

        def init(u, ud, udd):
            return u - k1 * ud + k2 * udd, ud - k1 * udd

        P = _svf_from(seg, Ts, wf, *init(p0, pd0, pdd0))
        s_d = math.cos(p0) * pd0
        s_dd = math.cos(p0) * pdd0 - math.sin(p0) * pd0**2
        S = _svf_from(np.sin(seg), Ts, wf, *init(math.sin(p0), s_d, s_dd))[:, 0]
        # sign of the TRUE rate: central difference of the raw angle (offline, non-causal).
        # Using the filtered rate would delay every sign switch by the filter group delay,
        # making the Coulomb regressor partly collinear with the viscous one (measured:
        # c came out NEGATIVE). Quantisation noise only matters near phi_d = 0.
        sgn = np.sign(np.gradient(seg, Ts))
        G = _svf_from(sgn, Ts, wf, sgn[0], 0.0)[:, 0]
        m = np.zeros(len(seg), bool)
        m[skip:] = True
        m &= np.abs(P[:, 1]) > min_rate
        cols = [-S, -P[:, 1], -G]
        if drag:
            cols.append(-P[:, 1] * np.abs(P[:, 1]))
        Xs.append(np.column_stack(cols)[m])
        Ys.append(P[m, 2])
    if not Xs:
        raise ValueError("no valid segments")
    X, y = np.vstack(Xs), np.concatenate(Ys)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, c, gamma = theta[:3]
    res = y - X @ theta
    return LSFit(math.sqrt(max(alpha, 1e-9)), float(c), float(gamma), float(np.sqrt(np.mean(res**2))), int(len(y)),
                 float(theta[3]) if drag else 0.0)


class OnlineOmegaRLS:
    """Scalar RLS with exponential forgetting for alpha = omega0^2 (see module doc).

    Inputs each control period: the MEASURED angle with the estimated zero
    bias removed (not the Kalman estimate: that one is propagated with the
    nominal model and would pull the estimate toward it), and the acceleration
    actually applied to the cart. The cart acceleration
    reaches the pendulum after the loop latency and the driver lag; the same
    delay/lag is applied to the command so both sides stay synchronous.
    """

    def __init__(self, p: PlantParams, Ts: float, latency: float = 0.0, actuator_tau: float = 0.0,
                 wf: float = 8.0, forget: float = 0.9997, min_excitation: float = 0.0, bounds=(0.6, 1.6),
                 warmup: float = 2.0,
                 reject_sigma: float = 3.0):
        self.p0, self.Ts = p, Ts
        self.g, self.c = p.g, p.viscous_c
        self.alpha0 = p.alpha
        self.alpha = p.alpha
        self.P = 1e3
        self.lam = forget
        self.min_exc = min_excitation
        self.lo, self.hi = bounds[0] * p.alpha, bounds[1] * p.alpha
        self.d = int(round(latency / Ts))
        self.buf: deque = deque([0.0] * (self.d + 1), maxlen=self.d + 1)
        self.k_lag = 1.0 - math.exp(-Ts / actuator_tau) if actuator_tau > 0 else 1.0
        self.a_act = 0.0
        self.f_th = SVF(Ts, wf)
        self.f_phi = SVF(Ts, wf)
        self.n = 0
        self.n_used = 0
        self.n_rejected = 0
        self.reject = reject_sigma
        self.warmup = warmup
        self.e2 = 0.0  # running residual variance

    def reset(self, theta0: float = 0.0) -> None:
        self.f_th.reset(theta0)
        self.f_phi.reset(0.0)
        self.n = 0

    def update(self, theta: float, a_applied: float) -> float:
        self.buf.append(a_applied)
        self.a_act += self.k_lag * (self.buf[0] - self.a_act)  # delayed, lagged cart acceleration
        phi = math.sin(theta) - math.cos(theta) * self.a_act / self.g
        f, fd, fdd = self.f_th.step(theta)
        phi_f, _, _ = self.f_phi.step(phi)
        self.n += 1
        # skip the start-up transient (bias estimate settling, saturation) and, optionally,
        # samples with tiny excitation. NOTE: selecting samples on |phi_f| biases the
        # estimate (phi_f is noisy) - measured: 5.24 instead of 5.55 rad/s - so it is off.
        if self.n < int(self.warmup / self.Ts) or abs(phi_f) < self.min_exc:
            return self.alpha
        y = fdd + self.c * fd
        e = y - self.alpha * phi_f
        # robust step: a residual far outside the running spread is an unmodelled
        # disturbance (hand tap, rail knock) - it would bias the estimate, skip it
        if self.n_used > int(1.0 / self.Ts) and e * e > self.reject**2 * max(self.e2, 1e-12):
            self.n_rejected += 1
            self.e2 = 0.999 * self.e2 + 0.001 * e * e
            return self.alpha
        k = self.P * phi_f / (self.lam + phi_f * self.P * phi_f)
        self.alpha = min(self.hi, max(self.lo, self.alpha + k * e))
        self.P = (self.P - k * phi_f * self.P) / self.lam
        self.P = min(self.P, 1e4)
        self.e2 = 0.995 * self.e2 + 0.005 * e * e
        self.n_used += 1
        return self.alpha

    @property
    def omega0(self) -> float:
        return math.sqrt(self.alpha)

    @property
    def omega0_std(self) -> float:
        """Approximate 1-sigma of omega0 from the RLS covariance."""
        var_alpha = self.P * max(self.e2, 1e-12)
        return math.sqrt(var_alpha) / (2 * math.sqrt(self.alpha))

    @property
    def converged(self) -> bool:
        return self.n_used > int(4.0 / self.Ts) and self.omega0_std < 0.015 * self.omega0


class AdaptiveUpdater:
    """Certainty-equivalence redesign from the online estimate (bounded, rate-limited).

    mode 'monitor' only estimates; mode 'update' swaps the controller law.
    """

    def __init__(self, p: PlantParams, cfg: dict, algorithm: str, mode: str = "monitor",
                 period: float = 2.0, max_step: float = 0.05, bounds=(0.8, 1.25),
                 excitation_amp: float = 0.03, excitation_period: float = 4.0, max_reject_fraction: float = 0.02):
        """mode: 'monitor' (estimate only) or 'update' (self-tuning).

        Safety rules for 'update' (from the twin experiments in docs/03 section 8):
          * a KNOWN excitation runs - x_ref square wave +-excitation_amp, period excitation_period;
            without it the estimate is biased by up to 8 %;
          * the RLS has converged (1-sigma < 1 % of omega0);
          * fewer than max_reject_fraction of the samples were rejected as disturbances
            (hand taps bias the estimate by up to -18 % while looking confident);
          * the new omega0 stays within bounds x nominal and moves at most max_step per update.
        """
        self.p_design, self.cfg, self.algo, self.mode = p, cfg, algorithm, mode
        self.p_nominal = p
        self.exc_amp, self.exc_period = excitation_amp, excitation_period
        self.max_reject = max_reject_fraction
        loop = cfg["loop"]
        self.rls = OnlineOmegaRLS(p, loop["Ts"], loop["latency_estimate"], loop["actuator_tau"])
        self.period, self.max_step = period, max_step
        self.lo, self.hi = bounds[0] * p.omega0, bounds[1] * p.omega0
        self.t_last = 0.0
        self.history: list[tuple[float, float, float, int]] = []  # (t, omega0_hat, std, redesigned)

    def reference(self, t: float) -> float:
        """Known excitation added to x_ref (square wave), 0 when disabled."""
        if self.exc_amp <= 0 or t < 1.0:
            return 0.0
        return self.exc_amp if int((t - 1.0) / (self.exc_period / 2)) % 2 == 0 else -self.exc_amp

    @property
    def reject_fraction(self) -> float:
        return self.rls.n_rejected / max(1, self.rls.n_used + self.rls.n_rejected)

    @property
    def trustworthy(self) -> bool:
        return self.rls.converged and self.exc_amp > 0 and self.reject_fraction < self.max_reject

    def step(self, t: float, theta: float, a_applied: float, controller, estimator=None):
        """Call once per control period. Returns a new controller when redesigned, else None."""
        self.rls.update(theta, a_applied)
        new_ctrl = None
        if t - self.t_last >= self.period:
            self.t_last = t
            w_hat = self.rls.omega0
            redesigned = 0
            if self.mode == "update" and self.trustworthy:
                w_cur = self.p_design.omega0
                target = min(self.hi, max(self.lo, w_hat))
                if abs(target - w_cur) / w_cur > 0.01:
                    step = max(-self.max_step, min(self.max_step, (target - w_cur) / w_cur))
                    w_new = w_cur * (1 + step)
                    self.p_design = replace(self.p_design, omega0=w_new)
                    new_ctrl = self._redesign(controller, estimator)
                    redesigned = 1
            self.history.append((t, w_hat, self.rls.omega0_std, redesigned))
        return new_ctrl

    def _redesign(self, controller, estimator):
        from ..config import make_controller

        c2 = make_controller(self.algo, self.p_design, self.cfg)
        # bumpless: keep internal controller states if shapes match
        if hasattr(controller, "q") and hasattr(c2, "q") and getattr(controller, "q").shape == c2.q.shape:
            c2.q = controller.q.copy()
        if hasattr(controller, "z") and hasattr(c2, "z"):
            c2.z = controller.z
        if estimator is not None and hasattr(estimator, "set_plant"):
            estimator.set_plant(self.p_design)
        return c2
