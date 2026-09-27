"""Closed-loop behaviour in the digital twin."""
import copy

import numpy as np
import pytest

from pendulum_lab.cli import _run_sim
from pendulum_lab.config import load_config

CFG = load_config()


@pytest.mark.parametrize("algo", ["pid", "lqr", "place"])
def test_balances_from_5_deg_with_kick(algo):
    r = _run_sim(algo, copy.deepcopy(CFG), kicks=[(5.0, 0.5)])
    m = r.metrics()
    assert m["balanced"], m
    assert m["rms_theta_last2s_deg"] < 0.2
    assert m["max_abs_x_cm"] < 16


def test_angle_only_pd_drifts_into_soft_limit():
    r = _run_sim("pd", copy.deepcopy(CFG))
    assert "soft limit" in r.trip_reason or "predicted stop" in r.trip_reason


@pytest.mark.parametrize("offset", [12, -24, 24])
def test_bias_state_compensates_zero_calibration_error(offset):
    c = copy.deepcopy(CFG)
    c["sim"]["calib_error"]["adc_upright_offset"] = offset
    c["sim"]["duration"] = 12
    r = _run_sim("lqr", c)
    assert r.balanced
    assert np.degrees(r["bias_hat"][-1]) == pytest.approx(-offset / CFG["sim"]["sensor"]["counts_per_rad"] * 57.2958, abs=0.1)
    assert abs(np.mean(r["x"][-400:])) < 0.01


def test_without_bias_state_the_cart_offset_follows_the_gain_ratio():
    c = copy.deepcopy(CFG)
    c["sim"]["calib_error"]["adc_upright_offset"] = 6
    c["estimator"]["estimate_bias"] = False
    c["sim"]["duration"] = 15
    r = _run_sim("lqr", c)
    delta = 6 / CFG["sim"]["sensor"]["counts_per_rad"]  # rad, measured angle is too large by delta
    from pendulum_lab.config import make_controller, plant_from

    ctl = make_controller("lqr", plant_from(c), c)
    ks = ctl.law().k_state
    # steady state: kx*x + kth*theta = 0 with theta_hat = theta + delta and theta ~ 0
    x_pred = -ks[2] * delta / ks[0]
    assert np.mean(r["x"][-400:]) == pytest.approx(-x_pred, rel=0.25)


def test_kalman_filter_reduces_command_noise():
    c1, c2 = copy.deepcopy(CFG), copy.deepcopy(CFG)
    c2["estimator"]["type"] = "derivative"
    a_kf = _run_sim("lqr", c1).metrics()["rms_accel"]
    a_dd = _run_sim("lqr", c2).metrics()["rms_accel"]
    assert a_kf < 0.5 * a_dd


def test_large_latency_destabilises_naive_but_not_delay_aware_lqr():
    base = copy.deepcopy(CFG)
    base["sim"]["driver"]["latency"] = 0.035
    base["sim"]["duration"] = 6
    aware = copy.deepcopy(base)
    aware["loop"]["latency_estimate"] = 0.041
    naive = copy.deepcopy(base)
    naive["controllers"]["lqr"].update(delay_aware=False, include_lag=False, r=0.05)
    naive["loop"]["latency_estimate"] = 0.041
    ra, rn = _run_sim("lqr", aware), _run_sim("lqr", naive)
    assert ra.balanced and ra.metrics()["rms_theta_last2s_deg"] < 0.2
    # linearly unstable (|z| > 1); saturation turns the divergence into a violent limit cycle
    mn = rn.metrics()
    assert (not rn.balanced) or (mn["rms_theta_last2s_deg"] > 1.0 and mn["rms_accel"] > 3.0)


def test_swingup_reaches_upright_inside_the_rail():
    c = copy.deepcopy(CFG)
    c["sim"]["duration"] = 15
    r = _run_sim("swingup", c)
    assert r.balanced, r.events
    assert np.max(np.abs(r["x"])) < 0.25
