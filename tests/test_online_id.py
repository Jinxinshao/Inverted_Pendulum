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
    t, phi = simulate_free_swing(5.55, c=0.02, gamma=0.1, a0=math.radians(88), duration=60, fs=200)
    sm, rng = SensorModel(glitch_prob=0.3), np.random.default_rng(0)  # junk while crossing the dead zone
    rows = [(tt, sm.adc_of(math.pi, rng)) for tt in np.arange(0, 3, 0.005)]
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
    a, b = fits["A"], fits["B"]
    assert a.omega0 == pytest.approx(5.55, rel=5e-4)
    assert b.omega0 == pytest.approx(5.55, rel=2e-3)   # SVF restarts in every dead-zone gap: ~0.1 % low
    # the truth is viscous + Coulomb: model selection must find it and recover both terms
    assert a.damping.name == "粘性+库仑"
    assert a.viscous_c == pytest.approx(0.02, abs=0.004)
    assert a.coulomb_gamma == pytest.approx(0.10, abs=0.015)
    # sensor self-calibration: the twin's true slope (period-amplitude relation), bottom
    # reading and wrap length (bottom-crossing timing)
    cal = a.calibration
    k_true = SensorModel().counts_per_rad
    assert cal.K == pytest.approx(k_true, abs=4 * a.stats["K_se"] + 0.5)
    assert cal.b == pytest.approx(1966.0 + math.pi * k_true, abs=3.0)
    assert cal.N == pytest.approx(2 * math.pi * k_true, abs=5.0)
    assert a.stats["n_glitch"] > 0  # glitches were injected and removed


def test_real_swing_recording_2026_09_27():
    """The first recording on the rig: release from ~91 deg, 60 s, dead zone at the bottom."""
    from pathlib import Path

    from pendulum_lab.cli import identify_csv

    path = Path(__file__).parent / "data" / "swing_20260927_151631.csv"
    fits = identify_csv(str(path), load_config(), out=lambda m: None)
    a, b = fits["A"], fits["B"]
    assert a.omega0 == pytest.approx(5.515, abs=0.002)
    assert abs(a.per_side["+"] - a.per_side["-"]) < 0.002       # both sides agree
    assert b.omega0 == pytest.approx(a.omega0, rel=2e-3)          # independent method agrees
    c = a.calibration.describe()
    assert c["counts_per_rad"] == pytest.approx(678.1, abs=3.0)
    assert c["implied_upright_adc"] == pytest.approx(1966.0, abs=10.0)  # plumb-line value of the rig
    assert c["adc_at_bottom"] > 4092                            # the true bottom is in the dead zone
    assert 13.0 < c["dead_zone_deg"] < 15.5
    assert a.damping.d > 0                                      # air drag is needed on this rod
    assert math.degrees(a.amp[0]) == pytest.approx(90.9, abs=1.0)


def test_foh_svf_has_no_sampling_bias():
    """(wf Ts)^2/12 feed-through bias of f_dd is corrected: alpha exact on noiseless data."""
    from pendulum_lab.control.online_id import svf_filter

    t, phi = simulate_free_swing(5.517, a0=math.radians(90), duration=15, fs=200)
    for wf in (20.0, 60.0):
        P = svf_filter(phi, 0.005, wf, hold="foh")
        S = svf_filter(np.sin(phi), 0.005, wf, hold="foh")
        m = t > 2
        alpha = -np.sum(P[m, 2] * S[m, 0]) / np.sum(S[m, 0] ** 2)
        assert alpha == pytest.approx(5.517**2, rel=3e-4)


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
    assert ad.p_design.omega0 == pytest.approx(5.0, rel=0.03)  # moved from the nominal 5.51


W_NOM = plant_from(load_config()).omega0


def test_hand_taps_are_detected_and_block_updates():
    r, ad = _adaptive_run(W_NOM, excitation=True, kicks=True)
    assert r.balanced
    assert ad.reject_fraction > 0.02 and not ad.trustworthy
    assert ad.p_design.omega0 == pytest.approx(W_NOM, abs=1e-6)


def test_no_excitation_no_update():
    r, ad = _adaptive_run(5.0, excitation=False)
    assert ad.p_design.omega0 == pytest.approx(W_NOM, abs=1e-6)
