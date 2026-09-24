"""Tkinter GUI: digital twin (simulation), analysis, and real-rig runtime.

Tabs
----
1. 数字孪生仿真  - pick algorithm/estimator, edit gains and non-idealities,
                  run in real time or slow motion, push the rod, move the target.
2. 闭环分析      - gains, closed-loop poles on the z-plane, delay margin,
                  effect of Ts and latency (linear theory next to the simulation).
3. 实物运行      - shadow / closed-loop run on the PD42S1 rig, or the fake rig
                  (software-in-the-loop) for classroom demonstration.
"""
from __future__ import annotations

import copy
import math
import time
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from ..config import ROOT, load_config, make_controller, make_estimator, plant_from, sim_config_from, swingup_overrides  # noqa: E402
from ..control.analysis import analyse  # noqa: E402
from ..sim.simulator import Simulation  # noqa: E402

ALGO_LABELS = {
    "lqr": "LQR 最优状态反馈",
    "pid": "串级 PID（角度内环+位置外环）",
    "place": "极点配置",
    "pd": "角度 PD（仅角度，小车会漂移）",
    "swingup": "能量起摆 + LQR 捕获（仅仿真）",
    "none": "无控制（开环）",
}
WINDOW_S = 10.0


class ParamForm(ttk.Frame):
    """Grid of labelled entries bound to a dict path in the config."""

    def __init__(self, master, fields: list[tuple[str, str, tuple]], cfg_getter):
        super().__init__(master)
        self.vars: dict[str, tuple[tk.StringVar, tuple, float]] = {}
        self.cfg_getter = cfg_getter
        for i, (key, label, path_scale) in enumerate(fields):
            path, scale = path_scale
            ttk.Label(self, text=label).grid(row=i, column=0, sticky="w", padx=2, pady=1)
            v = tk.StringVar()
            ttk.Entry(self, textvariable=v, width=9).grid(row=i, column=1, sticky="e", padx=2, pady=1)
            self.vars[key] = (v, path, scale)
        self.refresh()

    def refresh(self):
        cfg = self.cfg_getter()
        for v, path, scale in self.vars.values():
            d = cfg
            for k in path[:-1]:
                d = d[k]
            val = d[path[-1]]
            v.set(f"{val * scale:.6g}" if isinstance(val, (int, float)) and not isinstance(val, bool) else str(val))

    def apply(self, cfg: dict):
        for v, path, scale in self.vars.values():
            d = cfg
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = float(v.get()) / scale


class App(tk.Tk):
    def __init__(self, config_path: str | None = None):
        super().__init__()
        self.title("倒立摆计算机控制教学平台 - 数字孪生 / 实物运行")
        self.geometry("1400x900")
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.sim: Simulation | None = None
        self.running = False
        self.last_tick = time.perf_counter()
        self.hw_loop = None  # LoopProcess while a rig run is active

        self.algo = tk.StringVar(value="lqr")
        self.estimator = tk.StringVar(value=self.cfg["estimator"]["type"])
        self.bias = tk.BooleanVar(value=self.cfg["estimator"]["estimate_bias"])
        self.lqr_delay = tk.BooleanVar(value=True)
        self.lqr_lag = tk.BooleanVar(value=True)
        self.speed = tk.DoubleVar(value=1.0)
        self.x_ref = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="就绪")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.tab_sim = ttk.Frame(nb)
        self.tab_ana = ttk.Frame(nb)
        self.tab_hw = ttk.Frame(nb)
        nb.add(self.tab_sim, text="  数字孪生仿真  ")
        nb.add(self.tab_ana, text="  闭环分析  ")
        nb.add(self.tab_hw, text="  实物运行  ")
        self._build_sim_tab()
        self._build_analysis_tab()
        self._build_hw_tab()
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken").pack(fill="x", side="bottom")
        self.reset_sim()
        self.after(30, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================================================================ sim tab
    def _build_sim_tab(self):
        left = ttk.Frame(self.tab_sim, padding=6)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self.tab_sim)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="控制算法", font=("", 10, "bold")).pack(anchor="w")
        cb = ttk.Combobox(left, state="readonly", width=30, values=[f"{k}: {v}" for k, v in ALGO_LABELS.items()])
        cb.current(0)
        cb.pack(anchor="w", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: (self.algo.set(cb.get().split(":")[0]), self.reset_sim()))

        C = lambda: self.cfg  # noqa: E731
        box = ttk.LabelFrame(left, text="控制器参数", padding=4)
        box.pack(fill="x", pady=4)
        self.form_ctrl = ParamForm(box, [
            ("pd_wn", "PD ωn [rad/s]", (("controllers", "pd", "wn"), 1)),
            ("pd_z", "PD ζ", (("controllers", "pd", "zeta"), 1)),
            ("pid_wn", "PID 内环 ωn", (("controllers", "pid", "wn"), 1)),
            ("pid_z", "PID 内环 ζ", (("controllers", "pid", "zeta"), 1)),
            ("pid_wo", "PID 外环 ωo", (("controllers", "pid", "wo"), 1)),
            ("pid_zo", "PID 外环 ζo", (("controllers", "pid", "zeta_o"), 1)),
            ("pid_ki", "PID Ki", (("controllers", "pid", "ki"), 1)),
            ("q_x", "LQR q_x", (("controllers", "lqr", "q", 0), 1)),
            ("q_v", "LQR q_v", (("controllers", "lqr", "q", 1), 1)),
            ("q_th", "LQR q_θ", (("controllers", "lqr", "q", 2), 1)),
            ("q_om", "LQR q_ω", (("controllers", "lqr", "q", 3), 1)),
            ("r", "LQR r", (("controllers", "lqr", "r"), 1)),
        ], C)
        self.form_ctrl.pack(fill="x")
        ttk.Checkbutton(box, text="LQR/极点配置考虑延迟", variable=self.lqr_delay).pack(anchor="w")
        ttk.Checkbutton(box, text="LQR/极点配置考虑驱动器滞后", variable=self.lqr_lag).pack(anchor="w")
        ttk.Label(box, text="极点配置 (连续 s 平面, 逗号分隔):").pack(anchor="w")
        self.poles_var = tk.StringVar(value=", ".join(f"{re}{im:+}j" for re, im in self.cfg["controllers"]["place"]["poles"]))
        ttk.Entry(box, textvariable=self.poles_var, width=32).pack(anchor="w")

        box2 = ttk.LabelFrame(left, text="状态估计", padding=4)
        box2.pack(fill="x", pady=4)
        ttk.Radiobutton(box2, text="卡尔曼滤波", value="kalman", variable=self.estimator).pack(anchor="w")
        ttk.Radiobutton(box2, text="带滤波的微分", value="derivative", variable=self.estimator).pack(anchor="w")
        ttk.Checkbutton(box2, text="估计角度零点偏差", variable=self.bias).pack(anchor="w")

        box3 = ttk.LabelFrame(left, text="数字孪生（非理想因素）", padding=4)
        box3.pack(fill="x", pady=4)
        self.form_twin = ParamForm(box3, [
            ("th0", "初始角 θ0 [deg]", (("sim", "theta0_deg"), 1)),
            ("Ts", "采样周期 Ts [ms]", (("loop", "Ts"), 1000)),
            ("lat", "执行延迟 [ms]", (("sim", "driver", "latency"), 1000)),
            ("slat", "传感延迟 [ms]", (("sim", "sensor", "latency"), 1000)),
            ("tau", "驱动器滞后 τ [ms]", (("sim", "driver", "tau"), 1000)),
            ("noise", "ADC 噪声 [LSB]", (("sim", "sensor", "noise_counts"), 1)),
            ("off", "零点误差 [LSB]", (("sim", "calib_error", "adc_upright_offset"), 1)),
            ("bl", "皮带间隙 [mm]", (("sim", "driver", "backlash"), 1000)),
            ("cou", "库仑摩擦 γ", (("plant", "coulomb_gamma"), 1)),
            ("dlat", "设计假设延迟 [ms]", (("loop", "latency_estimate"), 1000)),
        ], C)
        self.form_twin.pack(fill="x")

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="应用并重置", command=self.reset_sim).grid(row=0, column=0, sticky="ew")
        self.run_btn = ttk.Button(btns, text="▶ 运行", command=self.toggle_run)
        self.run_btn.grid(row=0, column=1, sticky="ew")
        ttk.Button(btns, text="推杆 ←", command=lambda: self._kick(-0.6)).grid(row=1, column=0, sticky="ew")
        ttk.Button(btns, text="推杆 →", command=lambda: self._kick(+0.6)).grid(row=1, column=1, sticky="ew")
        ttk.Button(btns, text="单步", command=self._single_step).grid(row=2, column=0, sticky="ew")
        ttk.Button(btns, text="导出 CSV", command=self._export).grid(row=2, column=1, sticky="ew")
        ttk.Label(left, text="目标位置 x_ref [m]").pack(anchor="w")
        ttk.Scale(left, from_=-0.12, to=0.12, variable=self.x_ref, orient="horizontal").pack(fill="x")
        ttk.Label(left, text="播放速度 (慢动作教学)").pack(anchor="w")
        ttk.Scale(left, from_=0.1, to=2.0, variable=self.speed, orient="horizontal").pack(fill="x")

        self.fig = Figure(figsize=(9, 8), dpi=90)
        gs = self.fig.add_gridspec(4, 1, height_ratios=[1.6, 1, 1, 1])
        self.ax_anim = self.fig.add_subplot(gs[0])
        self.ax_th = self.fig.add_subplot(gs[1])
        self.ax_x = self.fig.add_subplot(gs[2], sharex=self.ax_th)
        self.ax_a = self.fig.add_subplot(gs[3], sharex=self.ax_th)
        self._init_plots(self.ax_anim, self.ax_th, self.ax_x, self.ax_a)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.info = tk.Text(right, height=5, font=("Consolas", 9))
        self.info.pack(fill="x")

    def _init_plots(self, ax_anim, ax_th, ax_x, ax_a, store=None):
        store = store if store is not None else self.__dict__
        ax_anim.set_xlim(-0.38, 0.38)
        ax_anim.set_ylim(-0.18, 0.45)
        ax_anim.set_aspect("equal")
        ax_anim.set_title("digital twin (x [m])", fontsize=9)
        ax_anim.plot([-0.35, 0.35], [0, 0], color="0.4", lw=3)
        rail = self.cfg["sim"]["rail_half"]
        soft = self.cfg["safety"]["x_soft"]
        for s in (-1, 1):
            ax_anim.axvline(s * rail, color="k", lw=2)
            ax_anim.axvline(s * soft, color="r", ls="--", lw=1)
        store["cart_patch"] = Rectangle((-0.03, -0.02), 0.06, 0.04, color="tab:blue")
        ax_anim.add_patch(store["cart_patch"])
        (store["rod_line"],) = ax_anim.plot([], [], lw=3, color="0.3")
        store["head"] = Circle((0, 0.38), 0.012, color="tab:orange")
        ax_anim.add_patch(store["head"])
        (store["ref_mark"],) = ax_anim.plot([0], [-0.05], "v", color="g")
        store["txt"] = ax_anim.text(-0.37, 0.40, "", fontsize=8, family="monospace", va="top")
        (store["l_th"],) = ax_th.plot([], [], lw=1, label="theta")
        (store["l_thm"],) = ax_th.plot([], [], lw=0.6, alpha=0.5, label="measured")
        ax_th.set_ylabel("theta [deg]")
        ax_th.legend(fontsize=7, loc="upper right")
        (store["l_x"],) = ax_x.plot([], [], lw=1, label="x")
        (store["l_xr"],) = ax_x.plot([], [], lw=0.8, ls="--", label="x_ref")
        ax_x.set_ylabel("x [cm]")
        ax_x.legend(fontsize=7, loc="upper right")
        (store["l_a"],) = ax_a.plot([], [], lw=0.8)
        ax_a.set_ylabel("a [m/s^2]")
        ax_a.set_xlabel("t [s]")
        for a in (ax_th, ax_x, ax_a):
            a.grid(alpha=0.3)

    def _gather_cfg(self) -> dict:
        cfg = copy.deepcopy(self.cfg)
        try:
            self.form_ctrl.apply(cfg)
            self.form_twin.apply(cfg)
            poles = []
            for s in self.poles_var.get().split(","):
                z = complex(s.strip().replace(" ", ""))
                poles.append([z.real, z.imag])
            cfg["controllers"]["place"]["poles"] = poles
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            raise
        cfg["estimator"]["type"] = self.estimator.get()
        cfg["estimator"]["estimate_bias"] = self.bias.get()
        for n in ("lqr", "place"):
            cfg["controllers"][n]["delay_aware"] = self.lqr_delay.get()
            cfg["controllers"][n]["include_lag"] = self.lqr_lag.get()
        cfg["sim"]["duration"] = 1e9
        return cfg

    def reset_sim(self):
        try:
            cfg = self._gather_cfg()
        except ValueError:
            return
        self.cfg = cfg
        name = self.algo.get()
        c = swingup_overrides(cfg) if name == "swingup" else cfg
        p = plant_from(c)
        sc = sim_config_from(c)
        if name == "swingup" and abs(sc.theta0_deg) < 90:
            sc.theta0_deg = 179.0
        try:
            ctl = make_controller(name, p, c)
            est = make_estimator(p, c, name)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("设计失败", f"{type(e).__name__}: {e}")
            return
        self.sim = Simulation(p, sc, ctl, est)
        self.running = False
        self.run_btn.config(text="▶ 运行")
        self._update_info(ctl, p, c)
        self._draw()
        self._update_analysis()

    def _update_info(self, ctl, p, c):
        self.info.delete("1.0", "end")
        d = ctl.describe()
        lines = [f"{ctl.label}"]
        for k, v in d.items():
            if k == "name":
                continue
            if isinstance(v, dict):
                lines.append(f"  {k}: " + ", ".join(f"{kk}={vv:.3g}" for kk, vv in v.items() if isinstance(vv, float)))
            elif isinstance(v, float):
                lines.append(f"  {k} = {v:.4g}")
            else:
                lines.append(f"  {k} = {v}")
        lines.append(f"  plant: ω0={p.omega0:.3f} rad/s, 不稳定极点 {p.unstable_pole:.2f} 1/s, L_eq={p.L_eq * 100:.1f} cm")
        self.info.insert("end", "\n".join(lines))

    def toggle_run(self):
        self.running = not self.running
        self.run_btn.config(text="⏸ 暂停" if self.running else "▶ 运行")
        self.last_tick = time.perf_counter()

    def _kick(self, v):
        if self.sim:
            self.sim.kick(v)

    def _single_step(self):
        if self.sim:
            self.sim.x_ref = self.x_ref.get()
            self.sim.step()
            self._draw()

    def _export(self):
        from tkinter import filedialog

        if not self.sim:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.sim.result().to_csv(path)
            self.status.set(f"已导出 {path}")

    # ================================================================ tick
    def _tick(self):
        now = time.perf_counter()
        dt = min(0.1, now - self.last_tick)
        self.last_tick = now
        if self.running and self.sim is not None:
            self.sim.x_ref = self.x_ref.get()
            n = max(1, int(round(dt * self.speed.get() / self.sim.cfg.Ts)))
            for _ in range(n):
                if not self.sim.step():
                    self.running = False
                    self.run_btn.config(text="▶ 运行")
                    break
            self._draw()
        if self.hw_loop is not None:
            self._draw_hw()
        self.after(40, self._tick)

    def _draw(self):
        s = self.sim
        if s is None:
            return
        L = s.log
        th, x = s.theta, s.x_c
        self._draw_cart(self.__dict__, x, th, s.x_ref)
        ev = s.events[-1] if s.events else ""
        self.txt.set_text(f"t={s.t:6.2f}s θ={math.degrees(th):+6.2f}° x={x * 100:+6.1f}cm\n{ev[:60]}")
        if L["t"]:
            t = np.asarray(L["t"])
            i0 = np.searchsorted(t, t[-1] - WINDOW_S)
            tt = t[i0:]
            self.l_th.set_data(tt, np.degrees(np.asarray(L["theta"][i0:])))
            self.l_thm.set_data(tt, np.degrees(np.asarray(L["theta_meas"][i0:])))
            self.l_x.set_data(tt, np.asarray(L["x"][i0:]) * 100)
            self.l_xr.set_data(tt, np.asarray(L["x_ref"][i0:]) * 100)
            self.l_a.set_data(tt, np.asarray(L["a_eff"][i0:]))
            for ax in (self.ax_th, self.ax_x, self.ax_a):
                ax.relim()
                ax.autoscale_view(scalex=False)
            self.ax_th.set_xlim(max(0.0, tt[-1] - WINDOW_S), max(WINDOW_S, tt[-1]))
        self.canvas.draw_idle()
        self.status.set(" | ".join(s.events[-2:]) if s.events else f"t = {s.t:.2f} s")

    @staticmethod
    def _draw_cart(st, x, th, x_ref, L_rod=0.38, L_low=0.12):
        st["cart_patch"].set_x(x - 0.03)
        xt, yt = x + L_rod * math.sin(th), L_rod * math.cos(th)
        xb, yb = x - L_low * math.sin(th), -L_low * math.cos(th)
        st["rod_line"].set_data([xb, xt], [yb, yt])
        st["head"].center = (xt, yt)
        st["ref_mark"].set_data([x_ref], [-0.05])

    # ============================================================ analysis
    def _build_analysis_tab(self):
        left = ttk.Frame(self.tab_ana, padding=6)
        left.pack(side="left", fill="both")
        self.ana_text = tk.Text(left, width=78, font=("Consolas", 9))
        self.ana_text.pack(fill="both", expand=True)
        ttk.Button(left, text="重新分析（使用左侧仿真页参数）", command=self._update_analysis).pack(fill="x")
        self.fig_ana = Figure(figsize=(6, 6), dpi=90)
        self.ax_z = self.fig_ana.add_subplot(111)
        self.canvas_ana = FigureCanvasTkAgg(self.fig_ana, master=self.tab_ana)
        self.canvas_ana.get_tk_widget().pack(side="left", fill="both", expand=True)

    def _update_analysis(self):
        cfg = self.cfg
        p = plant_from(cfg)
        Ts = cfg["loop"]["Ts"]
        lat = cfg["sim"]["sensor"]["latency"] + cfg["sim"]["driver"]["latency"]
        tau = cfg["sim"]["driver"]["tau"]
        out = [
            f"被控对象: ω0 = {p.omega0:.4f} rad/s, α = ω0² = {p.alpha:.3f} 1/s², β = α/g = {p.beta:.4f} 1/m",
            f"开环不稳定极点 p = {p.unstable_pole:.3f} 1/s (倍增时间 {math.log(2) / p.unstable_pole * 1e3:.0f} ms)",
            f"评估条件: Ts = {Ts * 1e3:.1f} ms, 总延迟 = {lat * 1e3:.1f} ms, 驱动器滞后 τ = {tau * 1e3:.1f} ms", "",
        ]
        ax = self.ax_z
        ax.clear()
        th = np.linspace(0, 2 * math.pi, 300)
        ax.plot(np.cos(th), np.sin(th), "k-", lw=0.8)
        for name in ("pd", "pid", "lqr", "place"):
            try:
                ctl = make_controller(name, p, cfg)
                rep = analyse(p, ctl.law(), Ts, lat, tau)
            except Exception as e:  # noqa: BLE001
                out.append(f"[{name}] 设计失败: {e}")
                continue
            out.append(f"[{ctl.label}]")
            out.extend("   " + ln for ln in rep.lines()[:6])
            if abs(rep.spectral_radius - 1) < 1e-6:
                out.append("   z=1 处双极点: 小车位置/速度不受控 → 必然漂移")
            out.append("")
            ax.plot(rep.z_poles.real, rep.z_poles.imag, "x", ms=7, label=name)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("closed-loop poles, z-plane (Ts, delay, lag included)", fontsize=9)
        self.canvas_ana.draw_idle()
        self.ana_text.delete("1.0", "end")
        self.ana_text.insert("end", "\n".join(out))

    # ============================================================ hardware
    def _build_hw_tab(self):
        left = ttk.Frame(self.tab_hw, padding=6)
        left.pack(side="left", fill="y")
        hw = self.cfg["hardware"]
        self.hw_sensor = tk.StringVar(value=hw["sensor_port"])
        self.hw_motor = tk.StringVar(value=hw["motor_port"])
        self.hw_mode = tk.StringVar(value="shadow")
        self.hw_algo = tk.StringVar(value="lqr")
        self.hw_dur = tk.StringVar(value="30")
        self.hw_state = tk.StringVar(value="未连接")
        for lbl, var in (("角度传感器串口", self.hw_sensor), ("电机驱动器串口", self.hw_motor), ("运行时长 [s]", self.hw_dur)):
            ttk.Label(left, text=lbl).pack(anchor="w")
            ttk.Entry(left, textvariable=var, width=14).pack(anchor="w")
        ttk.Label(left, text="算法").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.hw_algo, values=["lqr", "pid", "place", "pd"], state="readonly", width=12).pack(anchor="w")
        ttk.Label(left, text="模式").pack(anchor="w", pady=(6, 0))
        ttk.Radiobutton(left, text="影子模式（只读，不动电机）", value="shadow", variable=self.hw_mode).pack(anchor="w")
        ttk.Radiobutton(left, text="闭环（电机会运动！）", value="closed", variable=self.hw_mode).pack(anchor="w")
        ttk.Radiobutton(left, text="模拟实物（SIL 演示）", value="sil", variable=self.hw_mode).pack(anchor="w")
        ttk.Button(left, text="开始", command=self._hw_start).pack(fill="x", pady=(8, 2))
        tk.Button(left, text="停  止", bg="#c62828", fg="white", font=("", 16, "bold"), height=2,
                  command=self._hw_stop).pack(fill="x", pady=4)
        ttk.Label(left, textvariable=self.hw_state, wraplength=220, foreground="#1565c0").pack(anchor="w", pady=6)
        ttk.Label(left, wraplength=220, text=(
            "安全要求：闭环前必须完成 docs/05 的全部标定与检查；12 V 电源开关（急停）放在手边；"
            "停止按钮与空格键都会触发受控减速停车；若软件无响应，立即断开 12 V。")).pack(anchor="w")
        self.bind("<space>", lambda e: self._hw_stop())

        right = ttk.Frame(self.tab_hw)
        right.pack(side="left", fill="both", expand=True)
        self.fig_hw = Figure(figsize=(9, 8), dpi=90)
        gs = self.fig_hw.add_gridspec(4, 1, height_ratios=[1.6, 1, 1, 1])
        axs = [self.fig_hw.add_subplot(gs[0]), self.fig_hw.add_subplot(gs[1])]
        axs.append(self.fig_hw.add_subplot(gs[2], sharex=axs[1]))
        axs.append(self.fig_hw.add_subplot(gs[3], sharex=axs[1]))
        self.hw_axes = axs
        self.hw_art: dict = {}
        self._init_plots(*axs, store=self.hw_art)
        self.canvas_hw = FigureCanvasTkAgg(self.fig_hw, master=right)
        self.canvas_hw.get_tk_widget().pack(fill="both", expand=True)

    def _hw_start(self):
        if self.hw_loop is not None and self.hw_loop.alive:
            return
        from ..hw.process import LoopProcess
        from ..hw.runtime import calibration_from

        mode = self.hw_mode.get()
        algo = self.hw_algo.get()
        try:
            dur = float(self.hw_dur.get())
            cfg = copy.deepcopy(self.cfg)
            if mode in ("shadow", "closed"):
                calibration_from(cfg["hardware"])  # raises if not calibrated
            if mode == "closed":
                from ..hw.pd42s1 import Protocol

                miss = Protocol.load(ROOT / cfg["hardware"]["protocol_file"]).missing_for_motion()
                if miss:
                    raise RuntimeError(f"协议表未完成/未验证: {miss}（见 docs/07）")
                if not messagebox.askyesno("闭环确认", "小车将会运动。\n12 V 开关在手边？导轨上无障碍？扶好摆杆？"):
                    return
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("无法开始", str(e))
            return
        log = str(ROOT / "data" / f"gui_{algo}_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self.hw_loop = LoopProcess(cfg, mode, algo, dur, self.hw_sensor.get(), self.hw_motor.get(), log)
        self.hw_loop.start()
        self.hw_state.set("控制进程启动中…")

    def _hw_stop(self):
        if self.hw_loop is not None:
            self.hw_loop.stop()
            self.hw_state.set("停止请求已发送：受控减速中…")

    def _draw_hw(self):
        lp = self.hw_loop
        lp.poll()
        state, msg = lp.state, lp.message
        self.hw_state.set(f"[{state}] {msg}")
        rows = lp.rows[-2400:]
        if rows:
            a = np.array([[r[0], r[6], r[7], r[10], r[8], r[14]] for r in rows], float)
            t = a[:, 0] - a[0, 0]
            th = np.where(np.isnan(a[:, 3]), a[:, 1], a[:, 3])
            x = np.where(np.isnan(a[:, 4]), a[:, 2], a[:, 4])
            st = self.hw_art
            self._draw_cart(st, x[-1], th[-1], 0.0)
            st["txt"].set_text(f"{state}  θ={math.degrees(th[-1]):+6.2f}° x={x[-1] * 100:+6.1f}cm")
            st["l_th"].set_data(t, np.degrees(th))
            st["l_thm"].set_data(t, np.degrees(a[:, 1]))
            st["l_x"].set_data(t, x * 100)
            st["l_a"].set_data(t, a[:, 5])
            for ax in self.hw_axes[1:]:
                ax.relim()
                ax.autoscale_view(scalex=False)
            self.hw_axes[1].set_xlim(max(0.0, t[-1] - WINDOW_S), max(WINDOW_S, t[-1]))
            self.canvas_hw.draw_idle()
        if lp.done is not None and not lp.alive:
            self.hw_last = lp
            self.hw_loop = None
            self.status.set(f"实物运行结束: {lp.done[0]}；日志 {lp.done[2]}")

    def _on_close(self):
        if self.hw_loop is not None:
            self.hw_loop.stop()
            self.hw_loop.join(timeout=3.0)
        self.destroy()


def main(config_path: str | None = None):
    App(config_path).mainloop()


if __name__ == "__main__":
    main()
