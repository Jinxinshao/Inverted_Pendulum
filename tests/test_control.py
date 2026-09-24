"""Control-theory checks of the design and analysis code."""
import math

import numpy as np
import pytest

from pendulum_lab.control.analysis import analyse, closed_loop_matrix, delay_margin
from pendulum_lab.control.controllers import LQR, AnglePD, CascadePID, LinearLaw, PolePlacement
from pendulum_lab.control.discrete import ackermann, augment, discretize_with_delay
from pendulum_lab.model.dynamics import linear_model
from pendulum_lab.model.params import default_params

P = default_params()
TS = 0.005


def test_pd_threshold_kp_equals_g():
    """Continuous theory: angle PD is stable iff Kp > g (ideal actuator, tiny Ts)."""
    for kp, stable in ((0.9 * P.g, False), (1.3 * P.g, True)):
        law = LinearLaw(k_state=np.array([0, 0, kp, 3.0]))
        z = np.linalg.eigvals(closed_loop_matrix(P, law, 1e-4))
        pend = sorted(np.abs(z))[:2]  # the two pendulum poles (cart poles sit at z=1)
        assert (max(pend) < 1.0) == stable


def test_cascade_condition_kv_below_beta_kd_over_kp():
    c = CascadePID.design(P, TS)
    assert c.kp * c.kv < P.beta * c.kd
    bad = CascadePID(c.kp, 0.0, c.kd, c.kx, 1.2 * P.beta * c.kd / c.kp)
    assert not analyse(P, bad.law(), 1e-3, margin_search=False).stable


def test_fractional_delay_model_matches_integer_augmentation():
    A, B, _ = linear_model(P)
    dm = discretize_with_delay(A, B, TS, 2 * TS)
    assert np.allclose(dm.Gammas[0], 0) and np.allclose(dm.Gammas[1], 0)
    dm2 = discretize_with_delay(A, B, TS, 0.0)
    assert np.allclose(dm.Gammas[2], dm2.Gammas[0])
    half = discretize_with_delay(A, B, TS, 0.5 * TS)
    # the cart-velocity row integrates a pure input: both halves together give the full Ts
    assert np.allclose(half.Gammas[0][1] + half.Gammas[1][1], dm2.Gammas[0][1])
    assert half.Gammas[0][1] == pytest.approx(0.5 * TS)


def test_ackermann_places_requested_poles():
    A, B, _ = linear_model(P)
    dm = discretize_with_delay(A, B, TS, 0.012)
    F, G = augment(dm)
    want = np.array([0.95, 0.9, 0.97 + 0.02j, 0.97 - 0.02j] + [0.0] * (F.shape[0] - 4))
    K = ackermann(F, G, want)
    got = np.linalg.eigvals(F - G @ K)
    # a triple pole at z=0 is only reproduced to ~eps^(1/3) (repeated-root sensitivity)
    assert np.allclose(np.sort_complex(got), np.sort_complex(want), atol=1e-3)


def test_lqr_gain_signs_are_physical():
    K = LQR(P, TS, design_delay=0.0, actuator_tau=0.0).K.ravel()
    assert np.all(K[:4] < 0), "u = -Kx must push the cart toward the lean, all gains positive in u"


def test_delay_aware_lqr_increases_delay_margin():
    naive = LQR(P, TS, design_delay=0.0, actuator_tau=0.0)
    aware = LQR(P, TS, design_delay=0.02, actuator_tau=0.015)
    m_naive = delay_margin(P, naive.law(), TS, actuator_tau=0.015)
    m_aware = delay_margin(P, aware.law(), TS, actuator_tau=0.015)
    assert m_aware > m_naive + 0.01


@pytest.mark.parametrize("ctl", [
    CascadePID.design(P, TS), LQR(P, TS, design_delay=0.02, actuator_tau=0.015),
    PolePlacement(P, TS, design_delay=0.02, actuator_tau=0.015)])
def test_default_designs_tolerate_the_expected_latency(ctl):
    rep = analyse(P, ctl.law(), TS, delay=0.02, actuator_tau=0.015)
    assert rep.stable and rep.delay_margin > 0.04


def test_unstable_pole_value():
    assert P.unstable_pole == pytest.approx(math.sqrt(P.alpha), rel=0.01)
    assert P.unstable_pole == pytest.approx(5.53, abs=0.05)
