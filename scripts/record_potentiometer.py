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
  CSV 列 t_s（PC 时间，秒）、adc、mcu_t（若固件发送 T=... 时间戳）、line（原始文本行）
  控制台打印：采样率、时间间隔统计、下垂读数、噪声、最大摆幅（粗略）。
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

COUNTS_PER_DEG = 676.7 / 57.29578  # 本装置标定值（(4092-1966)/180°）


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


def stats(rows):
    if len(rows) < 3:
        return "样本太少"
    t = [r[0] for r in rows]
    a = [r[1] for r in rows]
    dt = sorted(t[i + 1] - t[i] for i in range(len(t) - 1))
    rate = (len(t) - 1) / (t[-1] - t[0])
    med = dt[len(dt) // 2] * 1e3
    p99 = dt[int(0.99 * (len(dt) - 1))] * 1e3
    mean = sum(a) / len(a)
    sd = (sum((x - mean) ** 2 for x in a) / len(a)) ** 0.5
    return (f"采样率 {rate:.1f} Hz, 间隔中位数 {med:.2f} ms, p99 {p99:.2f} ms, 最大 {dt[-1] * 1e3:.1f} ms; "
            f"ADC 均值 {mean:.1f}, 标准差 {sd:.2f}, 范围 [{min(a)}, {max(a)}]")


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
    ser.reset_input_buffer()
    rows: list = []
    print(f"已打开 {a.port} @ {a.baud}")
    input("① 让摆杆自然下垂并完全静止，然后按回车 ...")
    record(ser, a.rest, rows, "静止下垂")
    rest = list(rows)
    print("   静止: " + stats(rest))
    input("② 把杆拉到约 90°（水平）稳住，按回车后立即松手（不要推）...")
    t_release = time.perf_counter()
    record(ser, a.seconds, rows, "自由摆动")
    ser.close()

    t0 = rows[0][0] if rows else 0.0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "adc", "mcu_t", "line"])
        for r in rows:
            w.writerow([f"{r[0] - t0:.6f}", r[1], "" if r[2] is None else r[2], r[3]])
    swing = [r for r in rows if r[0] >= t_release]
    print("   摆动: " + stats(swing))
    if rest and swing:
        down = sum(r[1] for r in rest) / len(rest)
        valid = [r[1] for r in swing if 20 < r[1] < 4075]
        if valid:
            amp = max(abs(x - down) for x in valid) / COUNTS_PER_DEG
            print(f"   下垂读数 {down:.1f}，有效区内最大偏离约 {amp:.0f}°（下垂一侧有约 15° 电位器死区，属正常）")
    print(f"已保存 {out}（{len(rows)} 行）")
    print("分析：python -m pendulum_lab identify --csv " + out)


if __name__ == "__main__":
    main()
