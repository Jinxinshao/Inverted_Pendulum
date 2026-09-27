"""Free-swing identification against the reference ODE."""
import math

import numpy as np
import pytest

from pendulum_lab.model.identification import (
    extrema, fit_decay_summary, fit_free_decay, half_cycle_action, period_factor, simulate_free_swing,
)
from pendulum_lab.model.params import PendulumGeometry


def test_period_factor_known_values():
    assert period_factor(1e-6) == pytest.approx(1.0, abs=1e-9)
    assert period_factor(math.pi / 2) == pytest.approx(1.18034, rel=1e-4)  # 2K(1/2)/pi


def test_half_cycle_action_small_amplitude_limit():
    w0, A = 5.5, math.radians(2.0)
    assert half_cycle_action(A, w0 * w0)[0] == pytest.approx(math.pi * A * A * w0 / 2, rel=1e-3)


@pytest.mark.parametrize("model", ["viscous", "coulomb"])
def test_summary_fit_reproduces_stopwatch_measurement(model):
    fit = fit_decay_summary(90, 50, 20, 24.85, model=model)
    kw = {"c": fit.viscous_c} if model == "viscous" else {"gamma": fit.coulomb_gamma}
    t, phi = simulate_free_swing(fit.omega0, duration=25.5, **kw)
    te, ve = extrema(t, phi)
    pos = [(a, b) for a, b in zip(te, ve) if b > 0]
    t20, a20 = pos[19]  # 20th return to the start side
    assert t20 == pytest.approx(24.85, abs=0.02)
    assert math.degrees(a20) == pytest.approx(50.0, abs=0.3)


def test_measured_period_agrees_with_geometry():
    """Geometry (pivot 12 cm from the bottom) predicts T0 within 1 % of the swing test."""
    geo = PendulumGeometry().mass_properties()
    for model in ("viscous", "coulomb"):
        fit = fit_decay_summary(model=model)
        assert fit.period_small == pytest.approx(geo["period_small"], rel=0.01)


def test_full_decay_fit_separates_viscous_and_coulomb():
    t, phi = simulate_free_swing(5.55, c=0.02, gamma=0.10, duration=40)
    fit = fit_free_decay(t, phi)
    assert fit.omega0 == pytest.approx(5.55, rel=1e-3)
    assert fit.viscous_c == pytest.approx(0.02, rel=0.05)
    assert fit.coulomb_gamma == pytest.approx(0.10, rel=0.05)


def test_fit_tolerates_potentiometer_dead_zone():
    t, phi = simulate_free_swing(5.55, c=0.03, duration=30)
    phi = phi.copy()
    phi[(phi > math.radians(0.3)) & (phi < math.radians(15.3))] = np.nan  # the rig's dead zone
    fit = fit_free_decay(t, phi)
    assert fit.omega0 == pytest.approx(5.55, rel=2e-3)
    assert fit.viscous_c == pytest.approx(0.03, rel=0.1)
