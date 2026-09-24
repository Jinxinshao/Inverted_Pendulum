"""Identification methods: SVF consistency, offline LS, one-sided decay fit through the
full sensor chain (dead zone + wrap + quantisation), online RLS and self-tuning."""
import csv
import math
from dataclasses import replace

import numpy as np
import pytest

from pendulum_lab.config import load_config, make_controller, make_estimator, plant_from, sim_config_from
from pendulum_lab.control.online_id import AdaptiveUpdater, ls_free_swing, svf_filter
from pendulum_lab.model.identification import simulate_free_swing
from pendulum_lab.sim.simulator import Disturbance, SensorModel, Simulation


def test_svf_second_derivative_is_consistent():
    Ts, w = 0.002, 5.0
    t = np.arange(0, 10, Ts)
    P = svf_filter(np.sin(w * t), Ts, 30.0)
    m = t > 3
    assert np.polyfit(P[m, 0], P[m, 2], 1)[0] == pytest.approx(-w * w, rel=2e-3)


def test_ls_free_swing_recovers_all_three_parameters():
    t, phi = simulate_free_swing(5.55, c=0.02, gamma=0.1, duration=40)
    fit = ls_free_swing(t, phi)
    assert fit.omega0 == pytest.approx(5.55, rel=2e-3)
    assert fit.viscous_c == pytest.approx(0.02, abs=0.005)
    assert fit.coulomb_gamma == pytest.approx(0.10, abs=0.02)


@pytest.fixture
def rig_csv(tmp_path):
    """What record_potentiometer.py would produce on this rig: 3 s rest, release from 88 deg."""
    t, phi = simulate_free_swing(5.55, c=0.02, gamma=0.1, a0=math.radians(88), duration=60, fs=500)
    sm, rng = SensorModel(), np.random.default_rng(0)
    rows = [(tt, sm.adc_of(math.pi, rng)) for tt in np.arange(0, 3, 0.002)]
    rows += [(3 + tt, sm.adc_of(th + math.pi, rng)) for tt, th in zip(t, phi)]
    p = tmp_path / "swing.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "adc", "mcu_t", "line"])
        for a, b in rows:
            w.writerow([f"{a:.6f}", b, "", ""])
    return str(p)


def test_identify_csv_through_the_sensor_chain(rig_csv):
    from pendulum_lab.cli import identify_csv

    fits = identify_csv(rig_csv, load_config(), out=lambda m: None)
    a1, b = fits["one_sided"], fits["svf_ls"]
    for f in (a1, b):
        assert f.omega0 == pytest.approx(5.55, rel=3e-3)
    assert a1.viscous_c == pytest.approx(0.02, abs=0.004)
    assert a1.coulomb_gamma == pytest.approx(0.10, abs=0.01)
    # the two-sided fit is biased by the wrap offset of the far side - documented pitfall
    assert abs(fits["two_sided"].coulomb_gamma - 0.10) > 0.03


def _adaptive_run(w_true, excitation, kicks=False, mode="update"):
    cfg = load_config()
    cfg["sim"]["duration"] = 40
    p_nom = plant_from(cfg)
    sc = sim_config_from(cfg)
    if kicks:
        sc.disturbances = [Disturbance(t, "omega_kick", 0.3 * (-1) ** i) for i, t in enumerate(np.arange(3, 40, 3))]
    ad = AdaptiveUpdater(p_nom, cfg, "lqr", mode, excitation_amp=0.03 if excitation else 0.0)
    r = Simulation(replace(p_nom, omega0=w_true), sc, make_controller("lqr", p_nom, cfg), make_estimator(p_nom, cfg), ad).run()
    return r, ad


def test_online_rls_with_known_excitation_retunes_the_controller():
    r, ad = _adaptive_run(5.0, excitation=True)
    assert r.balanced
    assert ad.rls.omega0 == pytest.approx(5.0, rel=0.025)
    assert ad.p_design.omega0 == pytest.approx(5.0, rel=0.03)  # moved from 5.55


def test_hand_taps_are_detected_and_block_updates():
    r, ad = _adaptive_run(5.55, excitation=True, kicks=True)
    assert r.balanced
    assert ad.reject_fraction > 0.02 and not ad.trustworthy
    assert ad.p_design.omega0 == pytest.approx(5.5518, abs=1e-6)


def test_no_excitation_no_update():
    r, ad = _adaptive_run(5.0, excitation=False)
    assert ad.p_design.omega0 == pytest.approx(5.5518, abs=1e-6)
