"""独立的电位器录制程序（只需要 pyserial，不依赖本工程其它代码）。

用途：把摆杆拉到约 90° 松手，录制 60 s 自由摆动的 ADC 输出，保存为 CSV，
发给老师分析，或直接用本工程辨识：
    python -m pendulum_lab identify --csv 你的文件.csv

用法（Windows 命令行，在本文件所在目录或工程根目录）：
    python -m pip install pyserial
    python scripts/record_potentiometer.py --port COM12 --seconds 60

步骤：
  1. 关闭 VOFA+ / XCOM 等占用 COM12 的软件（一个串口同时只能被一个程序打开）。
  2. 运行本程序，按提示先让杆静止下垂 3 秒（记录下垂零点和噪声）。
  3. 听到提示后，把杆拉到水平（约 90°）附近，稳住，再干脆地松手（不要推）。
  4. 保持桌面和小车不动，等待录制结束（默认 60 s）。
输出：
  CSV 列 t（PC 时间，秒）、adc、mcu_t（若固件发送 T=... 时间戳）、line（原始文本行）、
  phase（rest = 静止下垂段，swing = 摆动段）
  控制台打印：采样率、时间间隔统计、下垂读数（中位数，抗跳变）、噪声、两侧最大摆幅。

2026-09-27 第一次实录后的修正：
  * 等待按回车期间 ADC 板仍在发数据，串口缓冲区会积压几秒的旧数据，随后被一次读出、
    打上同一个时间戳（CSV 里出现上千行 t 相同）。现在每段录制开始前先清空输入缓冲区。
  * 下垂位置正好在电位器端点：偶尔会读到另一端（例如 4091 中夹一个 4），
    旧版用均值/标准差统计，标准差被这一个点拉到 114；现在用中位数和 MAD。
  * 过死区时会出现 1~5 个任意值的毛刺（例如 "1 1 1029 2129 4091"），旧版把它们当成
    有效角度，算出“最大偏离 181°”；现在用 9 点滑动中位数剔除毛刺，并分两侧给出摆幅。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import datetime

try:
    import serial  # pyserial
except ImportError:  # pragma: no cover
    sys.exit("请先安装 pyserial:  python -m pip install pyserial")

RE_ADC = re.compile(rb"ADC\s*[=:]\s*(-?\d+)", re.I)
RE_T = re.compile(rb"\bT\s*[=:]\s*(\d+)", re.I)
RE_INT = re.compile(rb"^\s*(-?\d+)")

COUNTS_PER_DEG = 678.1 / 57.29578  # 本装置：2026-09-27 自由摆动辨识（周期-振幅关系）
TURN = 4096 * 360 / 345              # 电位器转一整圈对应的读数（345° 电气角，含死区）


def parse(line: bytes):
    m = RE_ADC.search(line) or RE_INT.match(line)
    if not m:
        return None, None
    t = RE_T.search(line)
    return int(m.group(1)), (int(t.group(1)) if t else None)


def record(ser, seconds: float, rows: list, label: str):
    buf = bytearray()
    t_end = time.perf_counter() + seconds
    last_print = 0.0
    n0 = len(rows)
    while time.perf_counter() < t_end:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        now = time.perf_counter()
        buf.extend(chunk)
        while True:
            i = buf.find(b"\n")
            if i < 0:
                break
            line = bytes(buf[:i]).strip()
            del buf[: i + 1]
            adc, mt = parse(line)
            if adc is not None:
                rows.append((now, adc, mt, line.decode(errors="replace")))
        if now - last_print > 1.0:
            last_print = now
            n = len(rows) - n0
            left = t_end - now
            adc_now = rows[-1][1] if rows else "-"
            print(f"\r  {label}: {n} 个样本, 当前 ADC={adc_now}, 剩余 {left:4.1f} s   ", end="", flush=True)
    print()


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def stats(rows):
    if len(rows) < 3:
        return "样本太少"
    t = [r[0] for r in rows]
    a = [r[1] for r in rows]
    dt = sorted(t[i + 1] - t[i] for i in range(len(t) - 1))
    rate = (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else float("nan")
    med_dt = dt[len(dt) // 2] * 1e3
    p99 = dt[int(0.99 * (len(dt) - 1))] * 1e3
    m = median(a)
    mad = 1.4826 * median([abs(x - m) for x in a])
    jumps = sum(abs(x - m) > 20 for x in a)
    return (f"采样率 {rate:.1f} Hz, 间隔中位数 {med_dt:.2f} ms, p99 {p99:.2f} ms, 最大 {dt[-1] * 1e3:.1f} ms; "
            f"ADC 中位数 {m:.1f}, 噪声(MAD) {mad:.2f}, 偏离 >20 的点 {jumps} 个, 范围 [{min(a)}, {max(a)}]")


def swing_amplitudes(adc, rest):
    """Largest deviation on each side [deg], dead-zone glitches removed (9-point median test)."""
    u = [rest + ((x - rest + TURN / 2) % TURN) - TURN / 2 for x in adc]   # unwrap around the rest reading
    lo_side = hi_side = 0.0
    for i in range(4, len(u) - 4):
        if not (8 <= adc[i] <= 4087):
            continue                                   # dead zone / track end
        if abs(u[i] - median(u[i - 4:i + 5])) > 40:
            continue                                   # glitch while the wiper crosses the gap
        d = (u[i] - rest) / COUNTS_PER_DEG
        lo_side, hi_side = min(lo_side, d), max(hi_side, d)
    return -lo_side, hi_side


def main():
    ap = argparse.ArgumentParser(description="录制电位器自由摆动数据")
    ap.add_argument("--port", default="COM12")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--rest", type=float, default=3.0, help="开始前静止下垂记录的秒数")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"swing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    try:
        ser = serial.Serial(a.port, a.baud, timeout=0.01)
    except serial.SerialException as e:
        sys.exit(f"打不开 {a.port}: {e}\n→ 端口号对吗？VOFA+/XCOM 是否已关闭？")
    rows: list = []
    print(f"已打开 {a.port} @ {a.baud}")
    input("① 让摆杆自然下垂并完全静止，然后按回车 ...")
    ser.reset_input_buffer()   # discard what piled up while waiting for Enter
    record(ser, a.rest, rows, "静止下垂")
    rest = list(rows)
    print("   静止: " + stats(rest))
    input("② 把杆拉到约 90°（水平）稳住，按回车后立即松手（不要推）...")
    ser.reset_input_buffer()
    n_rest = len(rows)
    record(ser, a.seconds, rows, "自由摆动")
    ser.close()

    t0 = rows[0][0] if rows else 0.0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "adc", "mcu_t", "line", "phase"])
        for i, r in enumerate(rows):
            w.writerow([f"{r[0] - t0:.6f}", r[1], "" if r[2] is None else r[2], r[3], "rest" if i < n_rest else "swing"])
    swing = rows[n_rest:]
    print("   摆动: " + stats(swing))
    if rest and len(swing) > 20:
        down = median([r[1] for r in rest])
        a_lo, a_hi = swing_amplitudes([r[1] for r in swing], down)
        print(f"   下垂读数 {down:.1f}；两侧最大摆幅约 {a_lo:.0f}° / {a_hi:.0f}°"
              "（粗略值，误差约 2°；精确值用下面的 identify 命令。下垂位置旁有约 14° 电位器死区，"
              "过最低点时读数跳变属正常）")
    print(f"已保存 {out}（{len(rows)} 行）")
    print("分析：python -m pendulum_lab identify --csv " + out)


if __name__ == "__main__":
    main()
