"""Guided PD42S1 self-test: answers the three open hardware questions safely.

    python -m pendulum_lab drivertest --motor COM11 --protocol modbus
    python -m pendulum_lab drivertest --sim            (rehearsal against the emulator)

Questions (from docs/05):
  Q1  does a 0 rpm speed command (F1) really stop the motor?
  Q2  Modbus odd-length data: where is the pad byte?
  Q3  do the left/right limit switches stop the motor in speed mode?
plus (optional) the belt travel per motor revolution.

Safety design
  * every motion is slow (default 20 rpm for Q1, 10 rpm for Q3: about 13 / 7 mm/s
    with a 40 mm/rev belt pulley) and short (<= 1 s for Q1, 4 s for Q3), starting
    from the middle of the rail: the cart never reaches a rail end;
  * the limit switches are pressed BY HAND while the cart is far from them;
  * Ctrl+C or typing q at any prompt -> speed 0, then the motor is disabled;
    the power switch stays the physical emergency stop;
  * everything (prompts, answers, measurements, every hex frame) is written to
    a log so the result can be sent for analysis.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..hw.pd42s1 import COUNTS_PER_REV, PD42S1, DriverError, hexs, speed_payload


class Abort(Exception):
    pass


@dataclass
class Motion:
    rpm: float
    t: list = field(default_factory=list)      # s since the command
    pos: list = field(default_factory=list)    # encoder counts
    rpm_read: list = field(default_factory=list)

    def stationary_spans(self, t_from: float = 0.3, min_len: float = 0.15, tol_counts: float = 40.0):
        """Time spans in which the encoder did not move although motion was commanded."""
        spans, start = [], None
        for i in range(1, len(self.t)):
            if self.t[i] < t_from:
                continue
            still = abs(self.pos[i] - self.pos[i - 1]) <= tol_counts * (self.t[i] - self.t[i - 1]) / 0.02
            if still and start is None:
                start = self.t[i - 1]
            elif not still and start is not None:
                if self.t[i - 1] - start >= min_len:
                    spans.append((start, self.t[i - 1]))
                start = None
        if start is not None and self.t[-1] - start >= min_len:
            spans.append((start, self.t[-1]))
        return spans


class DriverSelfTest:
    def __init__(self, drv: PD42S1, out=print, ask=input, log_dir: str = "data", mm_per_rev: float = 40.0,
                 rpm_move: float = 20.0, rpm_limit: float = 10.0, config_path: str | None = None):
        self.drv, self._out, self._ask = drv, out, ask
        self.mm_per_rev = mm_per_rev
        self.rpm_move, self.rpm_limit = rpm_move, rpm_limit
        self.config_path = config_path
        stamp = time.strftime("%Y%m%d_%H%M%S")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / f"driver_selftest_{stamp}.txt"
        self.frames_path = Path(log_dir) / f"driver_selftest_{stamp}_frames.csv"
        self.lines: list[str] = []
        self.results: dict = {}
        self.plus_side = "?"
        self.last_zero_ok = True

    # ------------------------------------------------------------ io
    def say(self, msg: str = "") -> None:
        self._out(msg)
        self.lines.append(msg)

    def ask(self, prompt: str) -> str:
        a = self._ask(prompt)
        self.lines.append(prompt + " > " + a)
        if a.strip().lower() in ("q", "quit", "exit"):
            raise Abort("user quit")
        return a.strip()

    def yes(self, prompt: str, default: bool = False) -> bool:
        a = self.ask(prompt + (" [Y/n] " if default else " [y/N] ")).lower()
        return default if a == "" else a in ("y", "yes", "是", "1")

    def mm(self, counts: float) -> float:
        return counts / COUNTS_PER_REV * self.mm_per_rev

    # ------------------------------------------------------------ motor helpers
    def zero(self) -> bool:
        """Explicit F1 with 0.0 rpm. Returns True if the driver accepted it."""
        try:
            self.drv.write(0xF1, speed_payload(0.0, 0))
            return True
        except DriverError:
            self.drv.write(0xF1, speed_payload(0.1, 0))   # manual range 0.1..6000 rpm
            return False

    def safe_stop(self, disable: bool = True) -> None:
        for name, fn in (("零速", self.zero), ("失能", self.drv.disable if disable else None)):
            if fn is None:
                continue
            try:
                fn()
                self.say(f"   [安全] 已发送{name}")
            except Exception as e:  # noqa: BLE001 - best effort
                self.say(f"   [安全] {name}失败: {e} -> 请关闭电源")

    def move(self, rpm: float, seconds: float, dt: float = 0.02) -> Motion:
        """Speed command, poll encoder + speed, then zero speed (always)."""
        m = Motion(rpm)
        t0 = time.perf_counter()
        self.drv.write(0xF1, speed_payload(rpm, 0))
        try:
            while True:
                now = time.perf_counter() - t0
                if now > seconds:
                    break
                m.t.append(now)
                m.pos.append(self.drv.read_position())
                m.rpm_read.append(self.drv.read_speed())
                time.sleep(max(0.0, dt - (time.perf_counter() - t0 - now)))
        finally:
            self.last_zero_ok = self.zero()
        return m

    def watch(self, seconds: float, dt: float = 0.02) -> Motion:
        m = Motion(0.0)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            m.t.append(time.perf_counter() - t0)
            m.pos.append(self.drv.read_position())
            m.rpm_read.append(self.drv.read_speed())
            time.sleep(dt)
        return m

    def state(self) -> dict:
        s = self.drv.read_system()
        try:
            s["run_state"] = self.drv.read_status()
        except DriverError:
            s["run_state"] = None
        return s

    # ------------------------------------------------------------ steps
    def step0_readonly(self) -> None:
        d = self.drv
        self.say("\n== 第 0 步：只读检查（电机不会动） ==")
        fw, hw = d.read_version()
        sysp, drv, hom = d.read_system(), d.read_drive(), d.read_homing()
        self.results["drive"] = drv
        self.say(f"   固件 V{fw:.1f} 硬件 V{hw:.1f}，协议 {d.protocol}，地址 {d.address}")
        self.say(f"   总线电压 {sysp['bus_voltage_V']:.2f} V，实时位置 {sysp['position']}，目标位置 {sysp['target_pos']}，"
                 f"位置误差 {sysp['pos_error']}，使能 {'是' if sysp['enabled'] else '否'}，堵转 {'是' if sysp['stalled'] else '否'}")
        self.say(f"   工作模式 {drv['work_mode']}={drv['work_mode_name']}，细分 {drv['microstep']}，堵转保护 "
                 f"{'开' if drv['stall_protect'] else '关'}（{drv['stall_current_mA']} mA），速度环 P/I/D "
                 f"{drv['vel_P']}/{drv['vel_I']}/{drv['vel_D']}")
        self.say(f"   回零参数：限位开关功能 {'开' if hom['limit_switches_on'] else '关'}，上电自动回零 "
                 f"{'开' if hom['auto_home_on_power'] else '关'}")
        plausible = 8.0 < sysp["bus_voltage_V"] < 60.0 and abs(sysp["pos_error"]) < COUNTS_PER_REV
        same = self.yes("   这几项（电压、实时位置、使能、工作模式）和官方上位机显示的一致吗？", default=plausible)
        en_31 = sysp["enabled"]
        en_2f = d.read_enabled()
        self.results["Q2"] = {
            "read_pad": "末尾补 0x00（官方日志 0x31：39 字节数据 + 1 个 00）",
            "decode_plausible": plausible, "user_confirms_values": same,
            "one_byte_read_consistent": en_31 == en_2f,
            "write_06": "单寄存器写：数值在低字节（官方日志 01 06 00 FA 00 01 = 失能）",
        }
        self.say(f"   单字节读取（0x2F 使能标志）与系统参数块（0x31 第 37 字节）一致：{'是' if en_31 == en_2f else '否'}")

    def step1_mode(self) -> None:
        d = self.drv
        self.say("\n== 第 1 步：工作模式 ==")
        drv = self.results["drive"]
        if drv["work_mode"] == 1:
            self.say("   已是 通信速度模式(1)。")
        else:
            self.say(f"   当前为 {drv['work_mode_name']}({drv['work_mode']})。本软件用 F1 速度指令，必须是 通信速度模式(1)。")
            self.say("   程序将：失能 → 设置模式 1 → 读回确认 → 清除状态 → 使能 → 速度 0。电机不会转动。")
            if not self.yes("   现在切换？"):
                raise Abort("工作模式未切换，后续测试无法进行")
            d.disable()
            d.set_mode(1)
            drv = d.read_drive()
            self.results["drive"] = drv
            self.say(f"   读回工作模式：{drv['work_mode_name']}({drv['work_mode']})")
            if drv["work_mode"] != 1:
                raise Abort("工作模式设置失败")
            if self.yes("   是否“保存设备参数”（掉电后仍为速度模式）？", default=True):
                d.save_params()
                self.say("   已保存。")
        d.clear_state()
        d.enable()
        accepted = self.zero()
        self.say(f"   已清除状态并使能；零速指令{'被接受' if accepted else '被拒绝（改发 0.1 rpm）'}。")

    def step2_zero_speed(self) -> None:
        self.say("\n== 第 2 步（问题 1）：发 0 rpm 电机能否停住 ==")
        self.say("   准备：摆杆自然下垂；小车推到导轨中间；手离开小车；另一只手放在电源开关旁。")
        self.say(f"   小车将以 {self.rpm_move:g} rpm 移动 0.8 s（约 {self.rpm_move / 60 * 0.8 * self.mm_per_rev:.0f} mm），"
                 "然后发 0 rpm 并观察 1.2 s；再反向走回来。")
        self.ask("   准备好后按回车开始（输入 q 退出）")
        res = {}
        for sgn, name in ((1, "正转(+)"), (-1, "反转(-)")):
            p0 = self.drv.read_position()
            m = self.move(sgn * self.rpm_move, 0.8)   # ends with the 0 rpm command
            accepted = self.last_zero_ok
            w = self.watch(1.2)
            p_end = w.pos[-1]
            late = [p for tt, p in zip(w.t, w.pos) if tt > 0.4]
            drift = max(late) - min(late) if late else 0
            moving = [tt for tt, p in zip(w.t, w.pos) if abs(p - p_end) > 20]
            t_stop = (max(moving) if moving else 0.0) + (w.t[0] if w.t else 0.0)
            ok = accepted and drift <= 40 and all(r == 0 for tt, r in zip(w.t, w.rpm_read) if tt > 0.4)
            res[name] = {"accepted": accepted, "moved_mm": self.mm(m.pos[-1] - p0) if m.pos else 0.0,
                         "stop_time_ms": 1e3 * t_stop, "drift_counts_after_0.4s": drift, "ok": ok}
            self.say(f"   {name}: 走了 {res[name]['moved_mm']:+.1f} mm（按 {self.mm_per_rev:g} mm/圈 估算）；"
                     f"0 rpm {'被接受' if accepted else '被拒绝'}；约 {res[name]['stop_time_ms']:.0f} ms 内停住；"
                     f"0.4 s 后漂移 {drift} 计数（{self.mm(drift):.3f} mm）→ {'通过' if ok else '未通过'}")
            if sgn > 0:
                side = self.ask("   刚才“正转(+)”时小车往哪边走？输入 L（左）或 R（右）").upper()
                self.plus_side = {"L": "左", "R": "右"}.get(side[:1], "?")
        self.results["Q1"] = res

    def _limit_case(self, sgn: int, switch: str) -> dict:
        direction = "正转(+)" if sgn > 0 else "反转(-)"
        side = self.plus_side if sgn > 0 else {"左": "右", "右": "左"}.get(self.plus_side, "?")
        target = side if switch == "ahead" else {"左": "右", "右": "左"}.get(side, "?")
        what = "小车正驶向的那一侧" if switch == "ahead" else "小车背离的那一侧"
        self.say(f"\n   -- {direction}（向{side}走），按【{what}={target}侧】限位开关 --")
        self.ask(f"   按回车后小车以 {self.rpm_limit:g} rpm 慢速移动 4 s；看到它开始动后，用手指按住{target}侧限位开关约 1 s 再松开")
        before = self.state()
        m = self.move(sgn * self.rpm_limit, 4.0)
        after = self.state()
        spans = m.stationary_spans()
        pressed_stop = bool(spans)
        resumed = bool(spans) and spans[-1][1] < m.t[-1] - 0.1
        user = self.yes("   你按下开关时，电机停了吗？")
        latched = (not after["enabled"]) or after["stalled"]
        r = {"direction": direction, "switch": f"{target}侧（{what}）", "auto_detected_stop": pressed_stop,
             "stop_spans_s": [(round(a, 2), round(b, 2)) for a, b in spans], "resumed_after_release": resumed,
             "user_saw_stop": user, "enabled_after": after["enabled"], "stalled_after": after["stalled"],
             "run_state_after": after.get("run_state"), "enabled_before": before["enabled"]}
        self.say(f"   程序检测：{'有停止区间 ' + str(r['stop_spans_s']) if pressed_stop else '全程在动（开关没有让电机停）'}"
                 + ("，松开后又继续走" if resumed else ""))
        if latched:
            self.say("   驱动器现在处于失能/堵转状态（锁存）。")
            if self.yes("   执行“清除状态 + 使能”恢复？", default=True):
                self.drv.recover()
                self.zero()
        return r

    def step3_limits(self) -> None:
        self.say("\n== 第 3 步（问题 3）：速度模式下限位开关能否让电机停 ==")
        self.say("   小车始终在导轨中间附近，不会真正开到限位处；由你用手指按下开关来模拟“撞到限位”。")
        self.say("   共 4 次：两个方向 ×（按驶向一侧 / 按背离一侧）。任何时候觉得不对，关电源或按 Ctrl+C。")
        if self.plus_side == "?":
            side = self.ask("   “正转(+)”时小车往哪边走？输入 L 或 R").upper()
            self.plus_side = {"L": "左", "R": "右"}.get(side[:1], "?")
        cases = []
        for sgn in (1, -1):
            for sw in ("ahead", "behind"):
                cases.append(self._limit_case(sgn, sw))
        hom = self.drv.read_homing()
        self.results["Q3"] = {"limit_function_on": hom["limit_switches_on"], "cases": cases}
        if not hom["limit_switches_on"] and not any(c["auto_detected_stop"] or c["user_saw_stop"] for c in cases):
            self.say("\n   限位开关功能当前为“关”，开关没有让电机停。")
            if self.yes("   临时打开驱动器的限位开关功能（不保存，测试完恢复为关）再测驶向一侧？"):
                self.drv.set_limit_switches(True)
                try:
                    extra = [self._limit_case(1, "ahead"), self._limit_case(-1, "ahead")]
                finally:
                    self.drv.set_limit_switches(False)
                    self.say("   已恢复：限位开关功能 关。")
                self.results["Q3"]["with_limit_function_on"] = extra

    def step4_belt(self) -> None:
        from .hwtools import cart_calibrate_interactive, update_config_file

        self.say("\n== 第 4 步（可选）：皮带标定——电机转一圈小车走多远 ==")
        if not self.yes("   现在做吗？需要一把钢尺和一小段胶带。"):
            return
        res = cart_calibrate_interactive(self.drv, self.rpm_move, 1.5, out=self.say, ask=self.ask)
        self.results["belt"] = res
        self.mm_per_rev = res["meters_per_rev"] * 1000
        if self.config_path:
            update_config_file(self.config_path, ["hardware", "cart"], res)
            self.say(f"   已写入 {self.config_path} → hardware.cart")

    # ------------------------------------------------------------ main
    def run(self, steps=("0", "1", "2", "3", "4")) -> dict:
        self.say("PD42S1 驱动器分步测试。每一步开始前都会等你按回车；输入 q 退出；Ctrl+C = 立即停止并失能。")
        ok = False
        try:
            if "0" in steps:
                self.step0_readonly()
            if "1" in steps:
                self.step1_mode()
            if "2" in steps:
                self.step2_zero_speed()
            if "3" in steps:
                self.step3_limits()
            if "4" in steps:
                self.step4_belt()
            ok = True
        except (KeyboardInterrupt, Abort) as e:
            self.say(f"\n中止：{e or 'Ctrl+C'}")
        except DriverError as e:
            self.say(f"\n通信错误：{e}")
        finally:
            disable = True
            if ok:
                try:
                    disable = self.yes("\n测试结束。让电机失能（手可以推动小车；推荐）？", default=True)
                except Abort:
                    pass
            self.safe_stop(disable=disable)
            self.summary()
            self.save()
        return self.results

    def summary(self) -> None:
        r = self.results
        self.say("\n================ 结论 ================")
        if "Q1" in r:
            q = r["Q1"]
            acc = all(v["accepted"] for v in q.values())
            ok = all(v["ok"] for v in q.values())
            self.say(f"问题 1（0 rpm）：驱动器{'接受' if acc else '拒绝'} 0 rpm；"
                     + ("两个方向都能停住，软件直接发 0 rpm 即可。" if ok else "没有完全停住 → 把日志发给老师分析；闭环前不要上电运行。"))
        if "Q2" in r:
            q = r["Q2"]
            self.say(f"问题 2（Modbus 补齐）：读回复 {q['read_pad']}；解码值{'合理' if q['decode_plausible'] else '异常'}，"
                     f"你{'确认' if q['user_confirms_values'] else '未确认'}与官方软件一致；单字节读取"
                     f"{'一致' if q['one_byte_read_consistent'] else '不一致'}。{q['write_06']}。")
        if "Q3" in r:
            q = r["Q3"]
            stops = [c for c in q["cases"] if c["auto_detected_stop"] or c["user_saw_stop"]]
            self.say(f"问题 3（限位开关）：驱动器限位功能{'开' if q['limit_function_on'] else '关'}；"
                     f"4 次中有 {len(stops)} 次开关让电机停下。")
            for c in q["cases"]:
                self.say(f"   {c['direction']} 按{c['switch']}：{'停' if (c['auto_detected_stop'] or c['user_saw_stop']) else '没停'}"
                         + ("，松开后继续" if c["resumed_after_release"] else "")
                         + ("，之后需要清除状态" if (not c["enabled_after"] or c["stalled_after"]) else ""))
            if not stops:
                self.say("   → 限位开关在速度模式下不起作用：只能依靠软件软限位（safety.x_soft）和驱动器堵转保护，"
                         "闭环前务必先完成皮带标定和中点设置。")
        if "belt" in r:
            self.say(f"皮带：{r['belt']['meters_per_rev'] * 1000:.2f} mm/圈，中点编码器值 {r['belt']['center_counts']}")
        self.say(f"日志：{self.log_path}\n收发帧：{self.frames_path}\n请把这两个文件发给老师。")

    def save(self) -> None:
        self.log_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        with open(self.frames_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "tx", "rx", "latency_ms", "error"])
            t0 = self.drv.records[0].t if self.drv.records else 0.0
            for rec in self.drv.records:
                w.writerow([f"{rec.t - t0:.4f}", hexs(rec.tx), hexs(rec.rx),
                            "" if rec.latency is None else f"{rec.latency * 1e3:.2f}", rec.error])
