"""Regenerate the figures used in docs/ (python examples/make_teaching_figures.py)."""
import copy
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pendulum_lab.cli import _run_sim  # noqa: E402
from pendulum_lab.config import load_config, make_controller, plant_from  # noqa: E402
from pendulum_lab.control.analysis import analyse, closed_loop_matrix  # noqa: E402
from pendulum_lab.model.identification import extrema, fit_decay_summary, period_factor, simulate_free_swing  # noqa: E402
from pendulum_lab.tools.plotting import plot_results  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CFG = load_config()
P = plant_from(CFG)


def fig_compare():
    res = {a: _run_sim(a, copy.deepcopy(CFG), kicks=[(5.0, 0.5)]) for a in ("pd", "pid", "lqr", "place")}
    plot_results(res, str(OUT / "compare.png"), "theta0 = 5 deg, kick +0.5 rad/s at t = 5 s")


def fig_delay_sweep():
    lat = np.linspace(0.0, 0.08, 81)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ("pid", "lqr", "place"):
        law = make_controller(name, P, CFG).law()
        rho = [np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(P, law, 0.005, d, 0.015)))) for d in lat]
        ax.plot(lat * 1e3, rho, label=name)
    c = copy.deepcopy(CFG)
    c["controllers"]["lqr"].update(delay_aware=False, include_lag=False)
    law = make_controller("lqr", P, c).law()
    rho = [np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(P, law, 0.005, d, 0.015)))) for d in lat]
    ax.plot(lat * 1e3, rho, "--", label="lqr (delay/lag NOT modelled)")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("total loop latency [ms]  (Ts = 5 ms, driver lag 15 ms)")
    ax.set_ylabel("spectral radius |z|max")
    ax.set_ylim(0.95, 1.08)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("stability vs latency: |z|max < 1 is stable")
    fig.tight_layout()
    fig.savefig(OUT / "delay_sweep.png", dpi=110)
    plt.close(fig)


def fig_estimators():
    c1, c2 = copy.deepcopy(CFG), copy.deepcopy(CFG)
    c2["estimator"]["type"] = "derivative"
    r1, r2 = _run_sim("lqr", c1), _run_sim("lqr", c2)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(r2["t"], r2["omega_hat"], lw=0.6, label="filtered derivative")
    ax[0].plot(r1["t"], r1["omega_hat"], lw=0.9, label="Kalman filter")
    ax[0].plot(r1["t"], r1["omega"], "k--", lw=0.8, label="true")
    ax[0].set_ylabel("omega [rad/s]")
    ax[0].set_ylim(-0.6, 0.6)
    ax[1].plot(r2["t"], r2["a_eff"], lw=0.6, label=f"derivative, rms {r2.metrics()['rms_accel']:.2f}")
    ax[1].plot(r1["t"], r1["a_eff"], lw=0.8, label=f"Kalman, rms {r1.metrics()['rms_accel']:.2f}")
    ax[1].set_ylabel("a [m/s^2]")
    ax[1].set_xlabel("t [s]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("12-bit quantisation (0.085 deg) -> rate noise: estimator matters")
    fig.tight_layout()
    fig.savefig(OUT / "estimators.png", dpi=110)
    plt.close(fig)


def fig_bias():
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for off in (0, 6, 12):
        for bias in (False, True):
            c = copy.deepcopy(CFG)
            c["sim"]["calib_error"]["adc_upright_offset"] = off
            c["estimator"]["estimate_bias"] = bias
            c["sim"]["duration"] = 12
            r = _run_sim("lqr", c)
            ls = "-" if bias else "--"
            lab = f"zero error {off / 678.1 * 57.3:.1f} deg, bias state {'on' if bias else 'off'}"
            ax[0].plot(r["t"], r["x"] * 100, ls, label=lab)
            if bias:
                ax[1].plot(r["t"], np.degrees(r["bias_hat"]), label=f"estimated bias, true {-off / 678.1 * 57.3:.2f} deg")
    ax[0].set_ylabel("x [cm]")
    ax[1].set_ylabel("bias_hat [deg]")
    ax[1].set_xlabel("t [s]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "bias.png", dpi=110)
    plt.close(fig)


def fig_swingup():
    c = copy.deepcopy(CFG)
    c["sim"]["duration"] = 12
    r = _run_sim("swingup", c)
    plot_results({"swing-up + LQR": r}, str(OUT / "swingup.png"), "energy swing-up from hanging (simulation only)")


def fig_free_swing():
    fv = fit_decay_summary(model="viscous")
    t, phi = simulate_free_swing(fv.omega0, c=fv.viscous_c, duration=25.5)
    te, ve = extrema(t, phi)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))
    ax[0].plot(t, np.degrees(phi), lw=0.7)
    ax[0].plot(te, np.degrees(ve), "o", ms=3)
    ax[0].set_ylabel("phi [deg]")
    ax[0].set_xlabel("t [s]")
    ax[0].set_title(f"fitted model reproduces 20 cycles 90->50 deg in 24.85 s (T0 = {fv.period_small:.4f} s)")
    A = np.linspace(1, 90, 200)
    ax[1].plot(A, period_factor(np.radians(A)))
    ax[1].set_xlabel("amplitude [deg]")
    ax[1].set_ylabel("T(A)/T0")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "free_swing.png", dpi=110)
    plt.close(fig)


def fig_real_swing():
    """First recording on the rig: raw ADC, cleaned angle, amplitude decay and period check."""
    from pendulum_lab.model.swing_analysis import analyse_swing
    from pendulum_lab.tools.hwtools import load_sensor_csv, split_rest_swing

    t, adc, ph = load_sensor_csv(str(OUT.parent.parent / "tests" / "data" / "swing_20260927_151631.csv"), with_phase=True)
    rest, sw = split_rest_swing(t, ph)
    r = analyse_swing(t[sw], adc[sw], float(np.median(adc[rest])))
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
    ts, a = t[sw], adc[sw]
    m = (ts > 49.9) & (ts < 51.4)
    ax[0, 0].plot(ts[m], a[m], ".-", ms=3, lw=0.5)
    ax[0, 0].set_title("raw ADC around two bottom passes: plateaus 4092 / 0..1 and glitches in between")
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("ADC")
    ax[0, 1].plot(r.t, np.degrees(r.phi), lw=0.5)
    ax[0, 1].plot(r.t_ext, np.degrees(r.side * r.amp), "o", ms=2.5)
    ax[0, 1].set_title("angle from the bottom after cleaning (gaps = dead zone), turning points")
    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("deg")
    for s_, mk in ((1, "o"), (-1, "s")):
        k = r.side == s_
        ax[1, 0].plot(r.t_ext[k], np.degrees(r.amp[k]), mk, ms=3, label=f"{'+' if s_ > 0 else '-'} side")
    tm_, pm_ = simulate_free_swing(r.omega0, r.viscous_c, r.coulomb_gamma, a0=float(r.amp[0]),
                                   duration=float(r.t_ext[-1] - r.t_ext[0]) + 0.5, fs=200, d=r.quad_d)
    te_, ve_ = extrema(tm_, pm_)
    ax[1, 0].plot(te_ + r.t_ext[0], np.degrees(np.abs(ve_)), "k-", lw=0.8,
                  label=f"model: c={r.viscous_c:.3f} 1/s, d={r.quad_d:.4f} 1/rad")
    tv_, pv_ = simulate_free_swing(r.omega0, 0.0519, 0.0, a0=float(r.amp[0]), duration=float(r.t_ext[-1] - r.t_ext[0]) + 0.5, fs=200)
    te2, ve2 = extrema(tv_, pv_)
    ax[1, 0].plot(te2 + r.t_ext[0], np.degrees(np.abs(ve2)), "r--", lw=0.8, label="best viscous-only model")
    ax[1, 0].set_title("amplitude decay: air drag is needed at large amplitude")
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("amplitude [deg]")
    ax[1, 0].legend(fontsize=8)
    per = r.t_ext[2:] - r.t_ext[:-2]
    aeff = 0.25 * (r.amp[:-2] + 2 * r.amp[1:-1] + r.amp[2:])
    ax[1, 1].plot(np.degrees(aeff), per, ".", ms=4, label="measured full periods")
    A = np.linspace(15, 95, 100)
    ax[1, 1].plot(A, r.T0 * period_factor(np.radians(A)), "k-", lw=0.8, label=f"T0 * 2K(sin^2(A/2))/pi, T0 = {r.T0:.4f} s")
    ax[1, 1].set_title(f"period vs amplitude -> omega0 = {r.omega0:.4f} rad/s, sensor {r.calibration.K:.1f} LSB/rad")
    ax[1, 1].set_xlabel("amplitude [deg]")
    ax[1, 1].set_ylabel("period [s]")
    ax[1, 1].legend(fontsize=8)
    for a_ in ax.flat:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "real_swing.png", dpi=110)
    plt.close(fig)


def fig_poles():
    fig, ax = plt.subplots(figsize=(6, 6))
    th = np.linspace(0, 2 * math.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=0.8)
    for name in ("pd", "pid", "lqr", "place"):
        rep = analyse(P, make_controller(name, P, CFG).law(), 0.005, 0.018, 0.015, margin_search=False)
        ax.plot(rep.z_poles.real, rep.z_poles.imag, "x", ms=8, label=f"{name} |z|max={rep.spectral_radius:.4f}")
    ax.set_xlim(0.6, 1.05)
    ax.set_ylim(-0.2, 0.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("closed-loop poles near z = 1 (Ts 5 ms, latency 18 ms, lag 15 ms)")
    fig.tight_layout()
    fig.savefig(OUT / "poles.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_compare, fig_delay_sweep, fig_estimators, fig_bias, fig_swingup, fig_free_swing, fig_real_swing, fig_poles):
        f()
        print("ok", f.__name__)
