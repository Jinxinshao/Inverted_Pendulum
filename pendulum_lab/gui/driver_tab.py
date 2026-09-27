"""驱动器 tab: the functions of the official PD42S1 upper computer that this
project needs, implemented from the V1.2 manuals, plus the belt calibration.

Usage rules (manual + practice):
  使能 before use; after 失能 enable again; 立即停止(刹车) latches -> 清除电机状态
  then 使能 before the motor moves again ("一键恢复" does both).
The tab owns the motor serial port only while "已连接"; the real-time run
closes it first (one port, one owner).
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk

from ..hw.pd42s1 import COUNTS_PER_REV, PD42S1, RUN_STATES, DriverError


class DriverTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        hw = app.cfg["hardware"]
        self.port = None
        self.drv: PD42S1 | None = None
        self.v_port = tk.StringVar(value=hw["motor_port"])
        self.v_baud = tk.StringVar(value=str(hw["motor_baud"]))
        self.v_proto = tk.StringVar(value=hw.get("motor_protocol", "custom"))
        self.v_addr = tk.StringVar(value=str(hw["motor_address"]))
        self.v_status = tk.StringVar(value="未连接")
        self.v_rpm = tk.StringVar(value="20")
        self.v_acc = tk.StringVar(value="10")
        self.v_auto = tk.BooleanVar(value=False)
        self.v_cal_rpm = tk.StringVar(value="20")
        self.v_cal_t = tk.StringVar(value="1.5")
        self.v_cal_mm = tk.StringVar(value="")
        self.v_cal_res = tk.StringVar(value="")
        self.cal_c0 = None
        self.cal_c1 = None
        self.jog_deadline = 0.0
        self.sys_vars: dict[str, tk.StringVar] = {}
        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self):
        left = ttk.Frame(self, padding=6)
        left.pack(side="left", fill="y")
        mid = ttk.Frame(self, padding=6)
        mid.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=6)
        right.pack(side="left", fill="both", expand=True)

        box = ttk.LabelFrame(left, text="串口连接", padding=4)
        box.pack(fill="x")
        for r, (lbl, var, vals) in enumerate((("串口", self.v_port, None), ("波特率", self.v_baud, ["115200", "256000", "460800", "921600"]),
                                               ("协议", self.v_proto, ["custom", "modbus"]), ("地址", self.v_addr, None))):
            ttk.Label(box, text=lbl).grid(row=r, column=0, sticky="w")
            if vals:
                ttk.Combobox(box, textvariable=var, values=vals, width=10).grid(row=r, column=1, sticky="w")
            else:
                ttk.Entry(box, textvariable=var, width=12).grid(row=r, column=1, sticky="w")
        ttk.Label(box, text="协议: custom=正点原子自定义, modbus=Modbus-RTU\n（与原厂上位机“设置协议类型”一致）",
                  foreground="gray").grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(box, text="连接", command=self.connect).grid(row=5, column=0, sticky="ew")
        ttk.Button(box, text="断开", command=self.disconnect).grid(row=5, column=1, sticky="ew")
        ttk.Label(box, textvariable=self.v_status, foreground="#1565c0", wraplength=230).grid(row=6, column=0, columnspan=2, sticky="w")

        box = ttk.LabelFrame(left, text="电机操作（同原厂上位机）", padding=4)
        box.pack(fill="x", pady=4)
        acts = [
            ("使能电机", lambda d: d.enable()), ("失能电机", lambda d: d.disable()),
            ("立即停止（刹车）", lambda d: d.brake()), ("清除电机状态", lambda d: d.clear_state()),
            ("一键恢复（清除+使能）", lambda d: d.recover()), ("清除当前位置", lambda d: d.zero_position()),
            ("设为通信速度模式", lambda d: d.set_mode(1)), ("打开指令回响", lambda d: d.set_echo(True)),
            ("左右限位 开", lambda d: d.set_limit_switches(True)), ("左右限位 关", lambda d: d.set_limit_switches(False)),
            ("保存参数", lambda d: d.save_params()),
        ]
        for i, (txt, fn) in enumerate(acts):
            ttk.Button(box, text=txt, command=lambda f=fn, t=txt: self.do(t, f)).grid(row=i // 2, column=i % 2, sticky="ew")
        ttk.Button(box, text="一键准备（速度模式+回响+恢复+零速）", command=self.prepare).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(box, text="就绪检查", command=self.check).grid(row=7, column=0, columnspan=2, sticky="ew")

        box = ttk.LabelFrame(left, text="点动（按住运行，松开即停，最长 3 s）", padding=4)
        box.pack(fill="x", pady=4)
        ttk.Label(box, text="速度 [rpm]").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.v_rpm, width=8).grid(row=0, column=1)
        ttk.Label(box, text="加减速 [r/s², 0-200]").grid(row=1, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.v_acc, width=8).grid(row=1, column=1)
        b1 = ttk.Button(box, text="◀ 反转")
        b2 = ttk.Button(box, text="正转 ▶")
        b1.grid(row=2, column=0, sticky="ew")
        b2.grid(row=2, column=1, sticky="ew")
        for b, sgn in ((b1, -1), (b2, 1)):
            b.bind("<ButtonPress-1>", lambda e, s=sgn: self.jog(s))
            b.bind("<ButtonRelease-1>", lambda e: self.jog(0))

        # ---- status
        box = ttk.LabelFrame(mid, text="系统参数 (0x31)", padding=4)
        box.pack(fill="x")
        rows = [("bus_voltage_V", "总线电压 [V]"), ("phase_current_mA", "相电流 [mA]"), ("speed_rpm", "实时转速 [rpm]"),
                ("position", "实时位置 [计数]"), ("x_cm", "小车位置 [cm]（需标定）"), ("pos_error", "位置误差"),
                ("enabled", "使能"), ("stalled", "堵转"), ("run_state", "运行状态"), ("work_mode", "工作模式")]
        for i, (k, lbl) in enumerate(rows):
            ttk.Label(box, text=lbl).grid(row=i, column=0, sticky="w")
            v = tk.StringVar(value="-")
            self.sys_vars[k] = v
            ttk.Label(box, textvariable=v, width=16).grid(row=i, column=1, sticky="w")
        ttk.Button(box, text="读取", command=self.read_status).grid(row=len(rows), column=0, sticky="ew")
        ttk.Checkbutton(box, text="自动刷新", variable=self.v_auto).grid(row=len(rows), column=1, sticky="w")

        box = ttk.LabelFrame(mid, text="皮带标定：电机转一圈小车走多远", padding=4)
        box.pack(fill="x", pady=6)
        txt = ("① 把小车推到导轨中部，用胶带标记位置 → 点“记录起点”\n"
               "② 点“正转运行”：以下列转速运行指定时间\n"
               "③ 用尺子量小车移动距离 (mm) 填入 → “计算”\n"
               "④ 点动把小车移回导轨中心 → “设为中心”并保存")
        ttk.Label(box, text=txt, justify="left").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(box, text="① 记录起点", command=self.cal_start).grid(row=1, column=0, sticky="ew")
        ttk.Label(box, text="rpm / 秒").grid(row=2, column=0, sticky="w")
        f = ttk.Frame(box)
        f.grid(row=2, column=1, sticky="w")
        ttk.Entry(f, textvariable=self.v_cal_rpm, width=5).pack(side="left")
        ttk.Entry(f, textvariable=self.v_cal_t, width=5).pack(side="left")
        ttk.Button(box, text="② 正转运行", command=self.cal_move).grid(row=1, column=1, sticky="ew")
        ttk.Label(box, text="实测距离 [mm]").grid(row=3, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.v_cal_mm, width=10).grid(row=3, column=1, sticky="w")
        ttk.Button(box, text="③ 计算", command=self.cal_compute).grid(row=4, column=0, sticky="ew")
        ttk.Button(box, text="④ 设为中心并保存", command=self.cal_center).grid(row=4, column=1, sticky="ew")
        ttk.Label(box, textvariable=self.v_cal_res, foreground="#2e7d32", wraplength=300).grid(row=5, column=0, columnspan=2, sticky="w")

        ttk.Label(right, text="指令日志（[发]/[收] 十六进制，与原厂上位机格式一致）").pack(anchor="w")
        self.log = tk.Text(right, font=("Consolas", 9), width=60)
        self.log.pack(fill="both", expand=True)
        ttk.Button(right, text="清空日志", command=lambda: self.log.delete("1.0", "end")).pack(anchor="e")
        self.after(300, self._poll)

    # ---------------------------------------------------------- connection
    def connect(self):
        from ..tools.hwtools import open_serial

        self.disconnect()
        try:
            self.port = open_serial(self.v_port.get(), int(self.v_baud.get()))
            self.drv = PD42S1(self.port, self.v_proto.get(), int(self.v_addr.get()), timeout=0.08, log=self._log)
            fw, hw_ = self.drv.read_version()
            self.v_status.set(f"已连接 {self.v_port.get()}：固件 V{fw:.1f} 硬件 V{hw_:.1f}")
            hw = self.app.cfg["hardware"]
            hw.update(motor_port=self.v_port.get(), motor_baud=int(self.v_baud.get()),
                      motor_protocol=self.v_proto.get(), motor_address=int(self.v_addr.get()))
            self.read_status()
        except Exception as e:  # noqa: BLE001
            self.v_status.set(f"连接失败: {e}")
            self.disconnect(keep_status=True)

    def disconnect(self, keep_status: bool = False):
        if self.port is not None:
            try:
                self.port.close()
            except Exception:  # noqa: BLE001
                pass
        self.port, self.drv = None, None
        if not keep_status:
            self.v_status.set("未连接")

    @property
    def connected(self) -> bool:
        return self.drv is not None

    def _log(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")
        if int(self.log.index("end").split(".")[0]) > 3000:
            self.log.delete("1.0", "1000.0")

    def do(self, label: str, fn):
        if not self.connected:
            messagebox.showwarning("未连接", "请先连接驱动器")
            return
        try:
            fn(self.drv)
            self.v_status.set(f"{label}：成功")
        except DriverError as e:
            self.v_status.set(f"{label}：失败 {e}")
        self.read_status()

    def prepare(self):
        def seq(d: PD42S1):
            d.set_mode(1)
            d.set_echo(True)
            d.recover()
            d.set_speed(0.0, 0)

        self.do("一键准备", seq)
        self.check()

    def check(self):
        if not self.connected:
            return
        ok, problems, _ = self.drv.readiness()
        self.v_status.set("就绪：可以闭环" if ok else "未就绪：" + "；".join(problems))

    # ---------------------------------------------------------- status
    def read_status(self):
        if not self.connected:
            return
        try:
            s = self.drv.read_system()
            mode = self.drv.read_drive()["work_mode_name"]
            state = RUN_STATES.get(self.drv.read_status(), "?")
        except DriverError as e:
            self.v_status.set(f"读取失败: {e}")
            return
        cart = self.app.cfg["hardware"]["cart"]
        x = "-"
        if cart.get("meters_per_rev") and cart.get("center_counts") is not None:
            x = f"{(cart.get('sign') or 1) * (s['position'] - cart['center_counts']) / COUNTS_PER_REV * cart['meters_per_rev'] * 100:+.2f}"
        vals = {"bus_voltage_V": f"{s['bus_voltage_V']:.2f}", "phase_current_mA": s["phase_current_mA"],
                "speed_rpm": s["speed_rpm"], "position": s["position"], "x_cm": x, "pos_error": s["pos_error"],
                "enabled": "是" if s["enabled"] else "否", "stalled": "是" if s["stalled"] else "否",
                "run_state": state, "work_mode": mode}
        for k, v in vals.items():
            self.sys_vars[k].set(str(v))

    def _poll(self):
        now = time.time()
        if self.connected and self.jog_deadline and now > self.jog_deadline:
            self.jog(0)
        if self.connected and self.v_auto.get():
            self.read_status()
        self.after(300, self._poll)

    # ---------------------------------------------------------- jog
    def jog(self, sgn: int):
        if not self.connected:
            return
        try:
            rpm = abs(float(self.v_rpm.get())) * sgn
            acc = int(float(self.v_acc.get()))
            self.drv.set_speed(rpm, acc)
            self.jog_deadline = time.time() + 3.0 if sgn else 0.0
        except (DriverError, ValueError) as e:
            self.v_status.set(f"点动失败: {e}")

    # ---------------------------------------------------------- belt calibration
    def cal_start(self):
        if not self.connected:
            return
        self.cal_c0 = self.drv.read_position()
        self.v_cal_res.set(f"起点计数 {self.cal_c0}")

    def cal_move(self):
        if not self.connected or self.cal_c0 is None:
            messagebox.showinfo("标定", "先连接并“记录起点”")
            return
        rpm, t = float(self.v_cal_rpm.get()), float(self.v_cal_t.get())
        if not messagebox.askyesno("标定", f"小车将以 {rpm} rpm 正转 {t} s（约走 {rpm / 60 * t * 40:.0f} mm，若带轮 40 mm/转），导轨是否足够？"):
            return
        self.drv.set_speed(rpm, 10)
        self.update()
        time.sleep(t)
        self.drv.set_speed(0.0, 10)
        time.sleep(0.4)
        self.cal_c1 = self.drv.read_position()
        turns = (self.cal_c1 - self.cal_c0) / COUNTS_PER_REV
        self.v_cal_res.set(f"电机转了 {turns:+.4f} 圈（计数 {self.cal_c1 - self.cal_c0:+d}）。现在用尺子量移动距离。")

    def cal_compute(self):
        try:
            mm = float(self.v_cal_mm.get())
            turns = (self.cal_c1 - self.cal_c0) / COUNTS_PER_REV
            mpr = mm / 1000.0 / abs(turns)
        except (TypeError, ValueError, ZeroDivisionError):
            messagebox.showinfo("标定", "先完成 ①②，并填入毫米数")
            return
        sign = 1 if turns > 0 else -1
        cart = self.app.cfg["hardware"]["cart"]
        cart.update(meters_per_rev=round(mpr, 6), sign=sign)
        self.v_cal_res.set(f"每转 {mpr * 1000:.2f} mm（GT2 20 齿应为 40.0 mm）；编码器方向 {'+' if sign > 0 else '-'}。"
                           f" 小车刚才移动的方向定义为 +x，请在导轨上标记！")

    def cal_center(self):
        if not self.connected:
            return
        cart = self.app.cfg["hardware"]["cart"]
        if not cart.get("meters_per_rev"):
            messagebox.showinfo("标定", "先完成 ③ 计算")
            return
        cart["center_counts"] = self.drv.read_position()
        path = self.app.save_hardware_config()
        self.v_cal_res.set(self.v_cal_res.get() + f"\n中心计数 {cart['center_counts']}，已保存到 {path}")
