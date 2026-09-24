"""参数辨识 tab: angle calibration helper, free-swing recording, identification
with two independent methods, and "apply to model" (all controllers are
redesigned from the new model - certainty equivalence)."""
from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class IdentTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        hw = app.cfg["hardware"]
        self.v_port = tk.StringVar(value=hw["sensor_port"])
        self.v_baud = tk.StringVar(value=str(hw["sensor_baud"]))
        self.v_secs = tk.StringVar(value="60")
        self.v_status = tk.StringVar(value="")
        self.v_cal = tk.StringVar(value="")
        self.rec_thread = None
        self.rec_data = None
        self.rec_phase = ""
        self.t = self.adc = None
        self.fits: dict = {}
        self.cal_vals: dict = {}
        self._build()

    def _build(self):
        left = ttk.Frame(self, padding=6)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        box = ttk.LabelFrame(left, text="角度传感器", padding=4)
        box.pack(fill="x")
        ttk.Label(box, text="串口").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.v_port, width=10).grid(row=0, column=1)
        ttk.Label(box, text="波特率").grid(row=1, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.v_baud, width=10).grid(row=1, column=1)
        ttk.Label(box, text="请先关闭 VOFA+/XCOM", foreground="gray").grid(row=2, column=0, columnspan=2, sticky="w")

        box = ttk.LabelFrame(left, text="① 角度标定助手", padding=4)
        box.pack(fill="x", pady=4)
        ttk.Button(box, text="记录下垂（静止 2 s）", command=lambda: self.cal_record("down")).pack(fill="x")
        ttk.Button(box, text="记录直立（铅垂线对准 2 s）", command=lambda: self.cal_record("up")).pack(fill="x")
        ttk.Button(box, text="记录向 +x 侧倾约 10°", command=lambda: self.cal_record("tilt")).pack(fill="x")
        ttk.Button(box, text="计算并保存标定", command=self.cal_save).pack(fill="x")
        ttk.Label(box, textvariable=self.v_cal, wraplength=240, foreground="#2e7d32").pack(anchor="w")

        box = ttk.LabelFrame(left, text="② 自由摆动辨识", padding=4)
        box.pack(fill="x", pady=4)
        ttk.Label(box, text="录制时长 [s]").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.v_secs, width=6).grid(row=0, column=1, sticky="w")
        ttk.Button(box, text="开始录制（先静止 3 s，再提示松手）", command=self.record).grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(box, text="载入 CSV（record_potentiometer.py 或 sensor-log）", command=self.load).grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(box, text="重新辨识", command=self.identify).grid(row=3, column=0, columnspan=2, sticky="ew")
        ttk.Label(box, textvariable=self.v_status, wraplength=240, foreground="#1565c0").grid(row=4, column=0, columnspan=2, sticky="w")

        box = ttk.LabelFrame(left, text="③ 应用到模型（所有控制器按新模型重新设计）", padding=4)
        box.pack(fill="x", pady=4)
        ttk.Button(box, text="采用方法 A1（推荐）", command=lambda: self.apply("one_sided")).pack(fill="x")
        ttk.Button(box, text="采用方法 B", command=lambda: self.apply("svf_ls")).pack(fill="x")
        ttk.Label(box, wraplength=240, justify="left", text=(
            "闭环中的在线辨识（RLS + 已知激励）在“数字孪生仿真”页和“实物运行”页的“在线辨识”选项里；"
            "原理与局限见 docs/03 第 8 节。")).pack(anchor="w")

        self.fig = Figure(figsize=(9, 7), dpi=90)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.text = tk.Text(right, height=11, font=("Consolas", 9))
        self.text.pack(fill="x")

    # ----------------------------------------------------------- calibration
    def _open(self):
        from ..tools.hwtools import open_serial

        return open_serial(self.v_port.get(), int(self.v_baud.get()))

    def cal_record(self, which: str):
        from ..tools.hwtools import average_adc

        try:
            port = self._open()
            try:
                mean, sd, n = average_adc(port, 2.0)
            finally:
                port.close()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("串口", str(e))
            return
        self.cal_vals[which] = mean
        names = {"down": "下垂", "up": "直立", "tilt": "+x 侧倾"}
        self.v_cal.set("  ".join(f"{names[k]}={v:.1f}" for k, v in self.cal_vals.items()) + f"  (std {sd:.2f}, n={n})")

    def cal_save(self):
        c = self.cal_vals
        if not {"down", "up", "tilt"} <= set(c):
            messagebox.showinfo("标定", "请依次记录下垂、直立、+x 侧倾")
            return
        cpr = abs(c["down"] - c["up"]) / math.pi
        sign = 1 if c["tilt"] > c["up"] else -1
        cal = self.app.cfg["hardware"]["calibration"]
        cal.update(adc_upright=round(c["up"], 2), counts_per_rad=round(cpr, 2), sign=sign)
        path = self.app.save_hardware_config()
        self.v_cal.set(f"直立 ADC {c['up']:.1f}，{cpr:.1f} LSB/rad（手册 680.1），符号 {sign:+d}；已保存 {path}")

    # ----------------------------------------------------------- recording
    def record(self):
        if self.rec_thread and self.rec_thread.is_alive():
            return
        try:
            port = self._open()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("串口", str(e))
            return
        secs = float(self.v_secs.get())

        def work():
            from ..hw.sensor import SensorReader

            rd = SensorReader(port, keep_seconds=secs + 10)
            rd.start()
            t0 = time.perf_counter()
            self.rec_phase = "rest"
            while time.perf_counter() - t0 < 3.0:
                time.sleep(0.05)
            self.rec_phase = "release"
            while time.perf_counter() - t0 < 3.0 + secs:
                time.sleep(0.05)
            rd.stop()
            rd.join(1.0)
            port.close()
            self.rec_data = rd.snapshot()
            self.rec_phase = "done"

        messagebox.showinfo("录制", "让杆自然下垂并完全静止，点确定后开始（先录 3 s 静止）")
        self.rec_data = None
        self.rec_thread = threading.Thread(target=work, daemon=True)
        self.rec_thread.start()
        self._poll_rec(secs)

    def _poll_rec(self, secs):
        ph = self.rec_phase
        if ph == "rest":
            self.v_status.set("静止下垂记录中…")
        elif ph == "release":
            self.v_status.set("现在把杆拉到约 90° 并松手！（不要推）录制中…")
            self.bell()
        if ph == "done":
            data = self.rec_data
            t0 = data[0][0] if data else 0.0
            self.t = np.array([d[0] - t0 for d in data])
            self.adc = np.array([d[1] for d in data], float)
            from ..tools.hwtools import save_sensor_csv

            import os

            os.makedirs("data", exist_ok=True)
            path = f"data/swing_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            save_sensor_csv(data, path)
            self.v_status.set(f"录制完成 {len(data)} 个样本，已保存 {path}")
            self.identify()
            return
        self.after(500 if ph != "release" else 3000, lambda: self._poll_rec(secs))

    def load(self):
        from ..tools.hwtools import load_sensor_csv

        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        self.t, self.adc = load_sensor_csv(path)
        self.v_status.set(f"已载入 {path}：{len(self.t)} 个样本")
        self.identify()

    # ----------------------------------------------------------- identify
    def identify(self):
        if self.t is None:
            return
        import csv
        import os
        import tempfile

        from ..cli import identify_csv
        from ..model.identification import extrema, fit_decay_summary_both
        from ..tools.hwtools import swing_angle_from_adc

        # identify_csv works on files -> write a temp copy (keeps one code path with the CLI)
        fd, tmp = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "adc"])
            w.writerows(zip(self.t, self.adc.astype(int)))
        lines: list[str] = []
        try:
            self.fits = identify_csv(tmp, self.app.cfg, out=lines.append)
        except Exception as e:  # noqa: BLE001
            lines.append(f"辨识失败: {e}（摆幅太小？录制太短？标定符号不对？）")
            self.fits = {}
        finally:
            os.remove(tmp)
        p = self.app.cfg["plant"]
        lines.append(f"当前模型: omega0={p['omega0']:.4f} rad/s, c={p['viscous_c']:.4f}, gamma={p['coulomb_gamma']:.4f}  ({p.get('source', '')})")
        v, c = fit_decay_summary_both()
        lines.append(f"秒表法(20 次 24.85 s): 粘性 {v.omega0:.4f} / 库仑 {c.omega0:.4f} rad/s")
        self.text.delete("1.0", "end")
        self.text.insert("end", "\n".join(lines))

        from ..cli import _calib_or_default

        phi = swing_angle_from_adc(self.adc, _calib_or_default(self.app.cfg))
        self.ax1.clear()
        self.ax1.plot(self.t, np.degrees(phi), lw=0.6)
        te, ve = extrema(self.t, phi)
        self.ax1.plot(te, np.degrees(ve), "o", ms=3)
        self.ax1.set_ylabel("phi from bottom [deg]  (gaps = dead zone)")
        self.ax1.grid(alpha=0.3)
        self.ax2.clear()
        if len(te):
            self.ax2.plot(te[ve > 0], np.degrees(ve[ve > 0]), "o", ms=3, label="+ side")
            self.ax2.plot(te[ve < 0], -np.degrees(ve[ve < 0]), "s", ms=3, label="- side (abs)")
        fo = self.fits.get("one_sided")
        side = self.fits.get("side", -1)
        sel = np.sign(ve) == side
        if fo is not None and sel.sum() > 2:
            from ..model.identification import simulate_free_swing

            t_s, a_s = te[sel][1], float(np.abs(ve[sel][1]))  # skip the release extremum (hand)
            ts, ps = simulate_free_swing(fo.omega0, fo.viscous_c, fo.coulomb_gamma, a0=a_s, duration=float(self.t[-1] - t_s), fs=200)
            tm, vm = extrema(ts, ps)
            self.ax2.plot(tm + t_s, np.degrees(np.abs(vm)), "k-", lw=0.8, label="model A1 (fitted side)")
        self.ax2.set_xlabel("t [s]")
        self.ax2.set_ylabel("amplitude [deg]")
        self.ax2.grid(alpha=0.3)
        self.ax2.legend(fontsize=8)
        self.canvas.draw_idle()

    def apply(self, key: str):
        f = self.fits.get(key)
        if f is None:
            messagebox.showinfo("应用", "没有该方法的结果")
            return
        p = self.app.cfg["plant"]
        old = p["omega0"]
        p.update(omega0=round(f.omega0, 5), viscous_c=round(f.viscous_c, 5), coulomb_gamma=round(f.coulomb_gamma, 5),
                 source=f"free-swing identification ({key})")
        path = self.app.save_hardware_config()
        self.app.reset_sim()
        messagebox.showinfo("应用", f"omega0 {old:.4f} -> {f.omega0:.4f} rad/s；控制器已按新模型重新设计。\n已保存到 {path}")
