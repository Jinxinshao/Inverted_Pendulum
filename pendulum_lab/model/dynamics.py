"""Nonlinear and linearised cart-pendulum models (acceleration input).

State of the physical plant:  s = [x, v, theta, omega]
Input:                        a  (cart acceleration actually realised by the belt)

    x_d     = v
    v_d     = a
    theta_d = omega
    omega_d = alpha sin(theta) - beta cos(theta) a - c omega - gamma sign(omega)

Design models
-------------
``linear_model`` returns the continuous (A, B) used for controller design.
Two optional augmentations make the design model match the real actuator path

* ``actuator_tau > 0`` - the driver's velocity loop is modelled as a first-order
  lag.  The host integrates the acceleration command into a velocity command
  v_c, so the model state becomes [x, v, theta, omega, v_c] with
      v_c_d = a_cmd,  v_d = (v_c - v)/tau,  omega_d = alpha theta - beta v_d - c omega
* input delay (added after discretisation, see ``discrete.augment_delay``).
"""
from __future__ import annotations

import math

import numpy as np

from .params import PlantParams

STATE_NAMES = ("x", "v", "theta", "omega")


def pendulum_accel(p: PlantParams, theta: float, omega: float, a: float, coulomb_eps: float = 1e-3) -> float:
    """Angular acceleration of the pendulum for a given cart acceleration."""
    fr = p.coulomb_gamma * math.tanh(omega / coulomb_eps) if p.coulomb_gamma else 0.0
    return p.alpha * math.sin(theta) - p.beta * math.cos(theta) * a - p.viscous_c * omega - fr


def rk4_pendulum(p: PlantParams, theta: float, omega: float, a: float, dt: float) -> tuple[float, float]:
    """One RK4 step of (theta, omega) with the cart acceleration held constant."""

    def f(th, om):
        return om, pendulum_accel(p, th, om, a)

    k1 = f(theta, omega)
    k2 = f(theta + 0.5 * dt * k1[0], omega + 0.5 * dt * k1[1])
    k3 = f(theta + 0.5 * dt * k2[0], omega + 0.5 * dt * k2[1])
    k4 = f(theta + dt * k3[0], omega + dt * k3[1])
    theta_n = theta + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    omega_n = omega + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return theta_n, omega_n


def energy_upright(p: PlantParams, theta: float, omega: float) -> float:
    """Pendulum energy per unit inertia, zero at the upright rest position [rad^2/s^2]."""
    return 0.5 * omega * omega + p.alpha * (math.cos(theta) - 1.0)


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def linear_model(p: PlantParams, actuator_tau: float = 0.0) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Continuous-time linearisation about the upright equilibrium.

    Returns (A, B, state_names). Coulomb friction is not linearisable and is
    left out (it only matters inside a +-gamma/alpha = +-0.4 deg band).
    """
    al, be, c = p.alpha, p.beta, p.viscous_c
    if actuator_tau <= 0.0:
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, al, -c],
            ]
        )
        B = np.array([[0.0], [1.0], [0.0], [-be]])
        return A, B, STATE_NAMES
    tau = actuator_tau
    # states: x, v, theta, omega, v_c ; input a_cmd = d(v_c)/dt
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0 / tau, 0.0, 0.0, 1.0 / tau],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, be / tau, al, -c, -be / tau],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    B = np.array([[0.0], [0.0], [0.0], [0.0], [1.0]])
    return A, B, STATE_NAMES + ("v_cmd",)
