"""Matplotlib figures for simulation results and hardware logs."""
from __future__ import annotations

import csv
import math

import numpy as np


def _mpl():
    import matplotlib

    if matplotlib.get_backend().lower() not in ("tkagg", "qtagg", "qt5agg", "macosx"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_results(results: dict, path: str, title: str = "") -> None:
    """results: {label: SimResult}. Four stacked panels: theta, x, a, v_cmd."""
    plt = _mpl()
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for label, r in results.items():
        t = r["t"]
        ax[0].plot(t, np.degrees(r["theta"]), label=label)
        ax[1].plot(t, r["x"] * 100, label=label)
        ax[2].plot(t, r["a_eff"], label=label, lw=0.8)
        ax[3].plot(t, r["v_cmd"], label=label)
    ax[0].set_ylabel("theta [deg]")
    ax[1].set_ylabel("x [cm]")
    ax[2].set_ylabel("a [m/s^2]")
    ax[3].set_ylabel("v_cmd [m/s]")
    ax[3].set_xlabel("t [s]")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(loc="upper right", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_poles(reports: dict, path: str) -> None:
    """z-plane pole map of several LoopReports."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 6))
    th = np.linspace(0, 2 * math.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=0.8)
    for label, rep in reports.items():
        z = rep.z_poles
        ax.plot(z.real, z.imag, "x", ms=8, label=f"{label} (|z|max={rep.spectral_radius:.3f})")
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("closed-loop poles (z-plane)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def read_log(path: str) -> dict[str, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        rd = csv.reader(f)
        head = next(rd)
        cols: list[list] = [[] for _ in head]
        for row in rd:
            for i, v in enumerate(row):
                cols[i].append(v)
    out = {}
    for h, c in zip(head, cols):
        try:
            out[h] = np.array([float(v) for v in c])
        except ValueError:
            out[h] = np.array(c)
    return out


def plot_hw_log(path_csv: str, path_png: str) -> dict:
    plt = _mpl()
    d = read_log(path_csv)
    act = d["state"] != "WAIT_ARM"
    t = d["t"]
    fig, ax = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    ax[0].plot(t, np.degrees(d["theta_meas"]), lw=0.6, label="theta_meas")
    ax[0].plot(t[act], np.degrees(d["theta_hat"][act]), lw=0.9, label="theta_hat")
    ax[0].set_ylabel("deg")
    ax[1].plot(t, d["x_meas"] * 100, lw=0.6, label="x_meas")
    ax[1].plot(t[act], d["x_hat"][act] * 100, lw=0.9, label="x_hat")
    ax[1].set_ylabel("cm")
    ax[2].plot(t, d["a_eff"], lw=0.6, label="a_eff")
    ax[2].set_ylabel("m/s^2")
    ax[3].plot(t, d["v_cmd"], lw=0.8, label="v_cmd")
    ax[3].set_ylabel("m/s")
    ax[4].plot(t, d["dt"] * 1e3, lw=0.5, label="loop dt")
    ax[4].plot(t, d["sensor_age"] * 1e3, lw=0.5, label="sensor age")
    ax[4].set_ylabel("ms")
    ax[4].set_xlabel("t [s]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)
    dt = d["dt"][1:] * 1e3
    return {
        "rows": len(t),
        "loop_dt_median_ms": float(np.median(dt)),
        "loop_dt_p99_ms": float(np.percentile(dt, 99)),
        "loop_dt_max_ms": float(np.max(dt)),
        "sensor_age_p99_ms": float(np.percentile(d["sensor_age"] * 1e3, 99)),
        "theta_rms_active_deg": float(np.degrees(np.sqrt(np.nanmean(d["theta_hat"][act] ** 2)))) if act.any() else float("nan"),
        "bias_hat_final_deg": float(np.degrees(d["bias_hat"][act][-1])) if act.any() else float("nan"),
    }
