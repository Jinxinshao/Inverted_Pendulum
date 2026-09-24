"""State estimators: angle rate and cart velocity from quantised measurements.

The potentiometer gives theta with a 12-bit quantum of ~1.48 mrad (0.085 deg).
Differentiating that at 200 Hz produces rate noise of +-0.3 rad/s per count -
the derivative gain alone would turn that into ~5 m/s^2 of command noise.
Two remedies are provided:

* :class:`DirtyDerivative` - first-order filtered derivative (the classical
  "D-term with filter" of industrial PID). Simple, model-free, adds phase lag.
* :class:`KalmanEstimator` - steady-state Kalman filter built on the SAME
  linear model the controllers use (it knows the commanded acceleration and
  the loop latency), optionally estimating a constant angle-zero bias b:
      theta_meas = theta + b
  b is observable because a biased balance point makes the cart accelerate
  (a = g*theta), which the cart-position measurement reveals.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..model.dynamics import linear_model
from ..model.params import PlantParams
from .discrete import discretize_with_delay, dlqe


@dataclass
class Estimate:
    x: float
    v: float
    theta: float
    omega: float
    bias: float = 0.0


class Estimator:
    name = "base"

    def reset(self, theta0: float = 0.0, x0: float = 0.0) -> None:
        raise NotImplementedError

    def update(self, theta_meas: float, x_meas: float, v_cmd: float, a_applied: float) -> Estimate:
        """``a_applied`` is the acceleration command issued at the previous step."""
        raise NotImplementedError


class DirtyDerivative(Estimator):
    """omega = s/(tau s + 1) theta (Tustin), v from the host velocity command."""

    name = "derivative"

    def __init__(self, Ts: float, cutoff_hz: float = 25.0, theta_filter_hz: float | None = None):
        self.Ts = Ts
        self.tau = 1.0 / (2 * math.pi * cutoff_hz)
        self.tf = None if not theta_filter_hz else 1.0 / (2 * math.pi * theta_filter_hz)
        self.reset()

    def reset(self, theta0: float = 0.0, x0: float = 0.0) -> None:
        self.prev_theta = theta0
        self.omega = 0.0
        self.theta_f = theta0
        self.first = True

    def update(self, theta_meas, x_meas, v_cmd, a_applied) -> Estimate:
        if self.first:
            self.prev_theta = self.theta_f = theta_meas
            self.first = False
        th = theta_meas
        if self.tf:
            k = self.Ts / (self.tf + self.Ts)
            self.theta_f += k * (theta_meas - self.theta_f)
            th = self.theta_f
        T, tau = self.Ts, self.tau
        # Tustin: omega_k = ((2tau - T) omega_{k-1} + 2 (th_k - th_{k-1})) / (2tau + T)
        self.omega = ((2 * tau - T) * self.omega + 2 * (th - self.prev_theta)) / (2 * tau + T)
        self.prev_theta = th
        return Estimate(x_meas, v_cmd, th, self.omega)


class KalmanEstimator(Estimator):
    """Kalman filter on [x, v, theta, omega (, bias)] with delayed known input.

    Time-varying (covariance recursion from P0) by default so that the zero
    bias is learned quickly; ``time_varying=False`` uses the steady-state gain.

    Noise model (all tunable):
      * process: white acceleration mismatch on the cart (sigma_a) and white
        angular-acceleration disturbance on the rod (sigma_alpha), bias random walk.
      * measurement: theta quantum^2/12 + electrical noise; x resolution.
    """

    name = "kalman"

    def __init__(
        self,
        p: PlantParams,
        Ts: float,
        delay: float = 0.0,
        sigma_theta: float = 1.5e-3,
        sigma_x: float = 2e-4,
        sigma_a: float = 0.5,
        sigma_alpha: float = 3.0,
        estimate_bias: bool = False,
        sigma_bias_rw: float = 2e-4,
        sigma_bias0: float = math.radians(3.0),
        time_varying: bool = True,
    ):
        self.p, self.Ts, self.delay = p, Ts, delay
        A, B, _ = linear_model(p, 0.0)
        dm = discretize_with_delay(A, B, Ts, delay)
        self.Phi = dm.Phi
        self.Gammas = [g.ravel() for g in dm.Gammas]
        n = 4
        # process-noise input matrix: cart accel noise and rod angular accel noise
        Gw = np.zeros((4, 2))
        Gw[1, 0] = Ts
        Gw[3, 0] = -p.beta * Ts
        Gw[3, 1] = Ts
        Qw = Gw @ np.diag([sigma_a**2, sigma_alpha**2]) @ Gw.T + 1e-12 * np.eye(4)
        H = np.array([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        F = self.Phi
        self.estimate_bias = estimate_bias
        if estimate_bias:
            n = 5
            F = np.eye(5)
            F[:4, :4] = self.Phi
            Qw5 = np.zeros((5, 5))
            Qw5[:4, :4] = Qw
            Qw5[4, 4] = sigma_bias_rw**2
            Qw = Qw5
            H = np.hstack([H, np.array([[1.0], [0.0]])])
            self.Gammas = [np.append(g, 0.0) for g in self.Gammas]
        self.F, self.H = F, H
        Rv = np.diag([sigma_theta**2, sigma_x**2])
        self.Qw, self.Rv = Qw, Rv
        self.L, self.P_ss = dlqe(F, H, Qw, Rv)
        self.time_varying = time_varying
        # Initial covariance. With a bias state only theta+b is measured at t=0, so
        # theta and b start equally uncertain; the cart dynamics (a = g*theta)
        # then separate them. Tested: +-2 deg zero error is learned before the
        # cart leaves +-15 cm (tests/test_simulation.py).
        p0 = [1e-6, 1e-3, (sigma_bias0 if estimate_bias else sigma_theta) ** 2, 0.5]
        if estimate_bias:
            p0.append(sigma_bias0**2)
        self.P0 = np.diag(p0)
        self.n = n
        self.hist: deque = deque([0.0] * len(self.Gammas), maxlen=len(self.Gammas))
        self.reset()

    def reset(self, theta0: float = 0.0, x0: float = 0.0) -> None:
        self.s = np.zeros(self.n)
        self.s[0], self.s[2] = x0, theta0
        self.hist = deque([0.0] * len(self.Gammas), maxlen=len(self.Gammas))
        self.first = True
        self.P = self.P0.copy()

    def set_plant(self, p: PlantParams) -> None:
        """Adaptive use: swap the model (omega0 changed) keeping state, covariance and input history."""
        A, B, _ = linear_model(p, 0.0)
        dm = discretize_with_delay(A, B, self.Ts, self.delay)
        self.p, self.Phi = p, dm.Phi
        g = [x.ravel() for x in dm.Gammas]
        if self.estimate_bias:
            g = [np.append(x, 0.0) for x in g]
            self.F[:4, :4] = self.Phi
        else:
            self.F = self.Phi
        if len(g) == len(self.Gammas):
            self.Gammas = g

    def update(self, theta_meas, x_meas, v_cmd, a_applied) -> Estimate:
        if self.first:
            self.s[:] = 0.0
            self.s[0], self.s[2] = x_meas, theta_meas
            self.first = False
        else:
            # predict: s_k = Phi s_{k-1} + sum_j Gamma_j u_{k-1-j}
            self.hist.appendleft(a_applied)
            s = self.F @ self.s
            for g, u in zip(self.Gammas, self.hist):
                s = s + g * u
            self.s = s
            if self.time_varying:
                self.P = self.F @ self.P @ self.F.T + self.Qw
        y = np.array([theta_meas, x_meas])
        if self.time_varying:
            # covariance form: converges from P0 (fast initial bias learning) to the steady-state gain
            S = self.H @ self.P @ self.H.T + self.Rv
            L = self.P @ self.H.T @ np.linalg.inv(S)
            self.P = (np.eye(self.n) - L @ self.H) @ self.P
        else:
            L = self.L
        self.s = self.s + L @ (y - self.H @ self.s)
        b = float(self.s[4]) if self.estimate_bias else 0.0
        return Estimate(float(self.s[0]), float(self.s[1]), float(self.s[2]), float(self.s[3]), b)


class SwingEstimator(Estimator):
    """Full-revolution angle estimator for swing-up.

    * unwraps the angle (a reading jump of ~2 pi is a wrap, not motion);
    * bridges the potentiometer dead zone: when ``in_deadzone`` is flagged by
      the caller, the angle is predicted with the last rate (theta += omega Ts);
    * filtered derivative for omega.
    """

    name = "swing"

    def __init__(self, Ts: float, cutoff_hz: float = 15.0):
        self.Ts = Ts
        self.tau = 1.0 / (2 * math.pi * cutoff_hz)
        self.reset()

    def reset(self, theta0: float = 0.0, x0: float = 0.0) -> None:
        self.theta = theta0
        self.omega = 0.0
        self.first = True
        self.in_deadzone = False

    def update(self, theta_meas, x_meas, v_cmd, a_applied) -> Estimate:
        if self.first:
            self.theta = theta_meas
            self.first = False
            return Estimate(x_meas, v_cmd, self.theta, 0.0)
        if self.in_deadzone:
            self.theta += self.omega * self.Ts
        else:
            prev = self.theta
            d = (theta_meas - prev + math.pi) % (2 * math.pi) - math.pi
            new = prev + d
            T, tau = self.Ts, self.tau
            self.omega = ((2 * tau - T) * self.omega + 2 * d) / (2 * tau + T)
            self.theta = new
        return Estimate(x_meas, v_cmd, self.theta, self.omega)


class HybridEstimator(Estimator):
    """SwingEstimator everywhere, Kalman filter near the upright position."""

    name = "hybrid"

    def __init__(self, swing: SwingEstimator, kalman: KalmanEstimator, switch_angle: float = math.radians(30.0)):
        self.swing, self.kf, self.switch = swing, kalman, switch_angle
        self.use_kf = False

    @property
    def in_deadzone(self) -> bool:
        return self.swing.in_deadzone

    @in_deadzone.setter
    def in_deadzone(self, v: bool) -> None:
        self.swing.in_deadzone = v

    def reset(self, theta0: float = 0.0, x0: float = 0.0) -> None:
        self.swing.reset(theta0, x0)
        self.kf.reset(theta0, x0)
        self.use_kf = False

    def update(self, theta_meas, x_meas, v_cmd, a_applied) -> Estimate:
        e = self.swing.update(theta_meas, x_meas, v_cmd, a_applied)
        th = (e.theta + math.pi) % (2 * math.pi) - math.pi
        if abs(th) < self.switch:
            if not self.use_kf:
                self.kf.reset(th, x_meas)
                self.use_kf = True
            k = self.kf.update(th, x_meas, v_cmd, a_applied)
            return k
        self.use_kf = False
        return e
