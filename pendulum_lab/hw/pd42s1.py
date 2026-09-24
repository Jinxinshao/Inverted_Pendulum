"""PD42S1 closed-loop stepper driver (ALIENTEK) - both serial protocols of the manual.

Source: "PD42S1 自定义串口协议 V1.2" and "PD42S1 modbus-rtu 协议 V1.2" (xlsx).
Every frame layout below was cross-checked byte-for-byte against the official
upper computer's command log (tests/test_pd42s1_protocol.py).

Custom protocol ("正点原子自定义协议")
    request : C5 | addr | code | data ... | sum8 | 5C
    reply   : C5 | addr | code | err | data ... | sum8 | 5C
    sum8 = (sum of all bytes before it) & 0xFF;  err 0x01 = OK,
    0xE1 short frame, 0xE2 bad header, 0xE3 bad tail, 0xE4 checksum,
    0xE5 unsupported code, 0xE6 illegal data.

Modbus-RTU ("modbus协议") - the register address is the custom function code
    read  : addr 04 00 code 00 nreg CRC      -> addr 04 nbytes data CRC
    write1: addr 06 00 code hi lo CRC        -> echo
    writeN: addr 10 00 code 00 nreg nbytes data CRC -> addr 10 00 code 00 nreg CRC
    exception: addr (func|0x80) errcode CRC (1 func, 2 address, 3 value, 4 device)
    CRC16/MODBUS (poly 0xA001, init 0xFFFF), LOW byte first on the wire
    (the manual's table says "CRC16_H, CRC16_L" but the logged frames are low-first).
    Odd data lengths are padded with a leading 0x00 byte.

Numbers: all multi-byte integers big-endian; float32 big-endian IEEE-754
(e.g. speed 50.0 rpm = 42 48 00 00). Position: 51200 counts per motor turn.

Operating rules from the manual and from use of the official software:
  * enable before use (FA 00 = ENABLE, FA 01 = DISABLE - note the inverted sense);
  * FC "immediate stop (brake)" latches; FB "clear state (stall, brake, disable)"
    must be sent, then enable again, before the motor moves again;
  * for this software the working mode must be 0x01 "communication speed mode"
    and speed commands use F1: direction u8 (0 fwd / 1 rev), accel u8 (r/s^2,
    0..200, 0 = step change), speed float32 rpm (0.1..6000).
"""
from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass

HEADER, TAIL = 0xC5, 0x5C
COUNTS_PER_REV = 51200

WORK_MODES = {
    0: "通信位置模式", 1: "通信速度模式", 2: "通信力矩模式", 3: "脉冲模式", 4: "脉宽位置模式",
    5: "脉宽速度模式", 6: "脉宽力矩模式", 7: "回零模式", 8: "开环速度模式", 9: "开环位置模式",
    10: "开环脉冲模式", 11: "IO启停模式",
}
RUN_STATES = {0: "停止", 1: "任务完成", 2: "正在运行", 3: "过载", 4: "堵转", 5: "欠压"}
CUSTOM_ERRORS = {0x01: "OK", 0xE1: "帧长度不足", 0xE2: "帧头错误", 0xE3: "帧尾错误", 0xE4: "校验和错误",
                 0xE5: "不支持的功能码", 0xE6: "数据不合法"}
MODBUS_ERRORS = {1: "不支持的功能码", 2: "非法地址", 3: "非法数据值/超出范围", 4: "从机设备故障"}

# Modbus: codes written with function 10H (multi-register); everything else uses 06H
MODBUS_MULTI = {0x63, 0x67, 0x73, 0x90, 0x91, 0x95, 0x98, 0xE0, 0xE1, 0xE2, 0xE4,
                0xF0, 0xF1, 0xF2, 0xF3, 0xF5, 0xF6, 0xF7}
# data length (bytes) returned by each read code (after the custom err byte)
READ_LEN = {0x20: 2, 0x21: 4, 0x22: 8, 0x23: 2, 0x24: 4, 0x25: 16, 0x26: 12, 0x27: 12, 0x28: 4,
            0x29: 2, 0x2A: 4, 0x2B: 4, 0x2C: 1, 0x2D: 1, 0x2E: 2, 0x2F: 1, 0x30: 1, 0x31: 39,
            0x32: 44, 0x94: 17, 0x96: 1}
# custom-protocol write replies that carry one extra byte after err
CUSTOM_WRITE_EXTRA = {0x01: 1, 0x62: 1, 0x6F: 1, 0xFA: 1}


class DriverError(IOError):
    def __init__(self, msg: str, code: int | None = None):
        super().__init__(msg)
        self.code = code


def sum8(data: bytes) -> int:
    return sum(data) & 0xFF


def crc16_modbus(data: bytes) -> int:
    c = 0xFFFF
    for b in data:
        c ^= b
        for _ in range(8):
            c = (c >> 1) ^ 0xA001 if c & 1 else c >> 1
    return c


def hexs(b: bytes) -> str:
    return " ".join(f"{x:02X}" for x in b)


def parse_hex(s: str) -> bytes:
    return bytes.fromhex(s.replace(",", " ").replace("0x", "").replace("0X", ""))


# ======================================================================= custom
def build_custom(address: int, code: int, data: bytes = b"") -> bytes:
    body = bytes([HEADER, address & 0xFF, code & 0xFF]) + data
    return body + bytes([sum8(body), TAIL])


# backwards-compatible name used by older tests/tools
build_frame = build_custom


@dataclass
class Frame:
    address: int
    func: int
    body: bytes  # custom: err + data ; modbus: payload between func and CRC
    raw: bytes

    @property
    def status(self) -> int | None:
        return self.body[0] if self.body else None


class FrameParser:
    """Incremental custom-protocol parser that resynchronises on garbage.

    A frame ends at the first 0x5C whose preceding byte is a valid sum8 of
    everything before it; a 0x5C inside the data fails that test (1/256 false
    accept, further filtered by address/code matching in the client).
    """

    def __init__(self, max_len: int = 64):
        self.buf = bytearray()
        self.max_len = max_len
        self.discarded = 0

    def feed(self, data: bytes) -> list[Frame]:
        self.buf.extend(data)
        out: list[Frame] = []
        while True:
            try:
                start = self.buf.index(HEADER)
            except ValueError:
                self.discarded += len(self.buf)
                self.buf.clear()
                return out
            if start:
                self.discarded += start
                del self.buf[:start]
            found = False
            for end in range(4, min(len(self.buf), self.max_len)):
                if self.buf[end] == TAIL and self.buf[end - 1] == sum8(self.buf[: end - 1]):
                    raw = bytes(self.buf[: end + 1])
                    out.append(Frame(raw[1], raw[2], raw[3:-2], raw))
                    del self.buf[: end + 1]
                    found = True
                    break
            if found:
                continue
            if len(self.buf) >= self.max_len:
                self.discarded += 1
                del self.buf[0]
                continue
            return out


# ======================================================================= modbus
def _with_crc(b: bytes) -> bytes:
    c = crc16_modbus(b)
    return b + bytes([c & 0xFF, c >> 8])


def _pad_even(data: bytes) -> bytes:
    return (b"\x00" + data) if len(data) % 2 else data


def build_modbus_write(address: int, code: int, data: bytes) -> bytes:
    if not data:
        data = b"\x01"  # manual: "fixed data 0x01, only completes the frame"
    if code in MODBUS_MULTI:
        d = _pad_even(data)
        return _with_crc(bytes([address, 0x10, 0x00, code, 0x00, len(d) // 2, len(d)]) + d)
    d = _pad_even(data)
    if len(d) != 2:
        raise ValueError(f"code 0x{code:02X}: 06H write carries exactly one register")
    return _with_crc(bytes([address, 0x06, 0x00, code]) + d)


def build_modbus_read(address: int, code: int) -> bytes:
    n = (READ_LEN[code] + 1) // 2
    return _with_crc(bytes([address, 0x04, 0x00, code, 0x00, n]))


def modbus_expected_len(req: bytes) -> int:
    f = req[1]
    if f == 0x04:
        return 5 + 2 * req[5]
    return 8  # 06 echo, 10 ack


def check_modbus_reply(req: bytes, rx: bytes) -> tuple[bytes | None, str]:
    """Return (payload, error). payload: data for 04, echo payload for 06/10."""
    addr, func = req[0], req[1]
    for i in range(len(rx)):
        if rx[i] != addr:
            continue
        if i + 5 <= len(rx) and rx[i + 1] == (func | 0x80):
            fr = rx[i:i + 5]
            if crc16_modbus(fr[:3]) == fr[3] | (fr[4] << 8):
                return None, f"modbus exception {fr[2]}: {MODBUS_ERRORS.get(fr[2], '?')}"
        n = modbus_expected_len(req)
        if i + n <= len(rx) and rx[i + 1] == func:
            fr = rx[i:i + n]
            if crc16_modbus(fr[:-2]) == fr[-2] | (fr[-1] << 8):
                if func == 0x04:
                    return fr[3:-2], ""
                if func == 0x06 and fr != req:
                    return None, "06 echo differs from request"
                if func == 0x10 and fr[:6] != req[:6]:
                    return None, "10 ack differs from request"
                return fr[2:-2], ""
    return None, "incomplete"


# ================================================================ data helpers
def f32(b: bytes) -> float:
    return struct.unpack(">f", b)[0]


def i16(b: bytes) -> int:
    return struct.unpack(">h", b)[0]


def i32(b: bytes) -> int:
    return struct.unpack(">i", b)[0]


def u32(b: bytes) -> int:
    return struct.unpack(">I", b)[0]


def speed_payload(rpm: float, accel: int) -> bytes:
    """F1 payload: direction, accel (r/s^2, 0..200, 0 = step), |speed| float32 rpm."""
    direction = 1 if rpm < 0 else 0
    accel = int(max(0, min(200, accel)))
    return bytes([direction, accel]) + struct.pack(">f", abs(float(rpm)))


def decode_system(d: bytes) -> dict:
    """0x31 read system parameters (39 bytes)."""
    return {
        "bus_voltage_V": f32(d[0:4]), "phase_current_mA": i16(d[4:6]), "flux_mWb": f32(d[6:10]),
        "phase_R_ohm": f32(d[10:14]), "phase_L_mH": f32(d[14:18]), "speed_rpm": i16(d[18:20]),
        "target_pos": i32(d[20:24]), "position": i32(d[24:28]), "pos_error": i32(d[28:32]),
        "pulse_count": u32(d[32:36]),
        # manual: byte 37 "0: enabled 1: disabled" for 0x31 (0x2F uses the opposite sense)
        "enabled": d[36] == 0, "in_position": bool(d[37]), "stalled": bool(d[38]),
    }


def decode_drive(d: bytes) -> dict:
    """0x32 read drive parameters (44 bytes)."""
    return {
        "work_mode": d[0], "work_mode_name": WORK_MODES.get(d[0], "?"), "echo": d[1] == 0,
        "baud": u32(d[2:6]), "can_kbps": struct.unpack(">H", d[6:8])[0], "dir_level": d[8], "en_level": d[9],
        "microstep": struct.unpack(">H", d[10:12])[0], "pos_max_current_mA": i16(d[12:14]),
        "pos_P": u32(d[14:18]), "pos_I": u32(d[18:22]), "pos_D": u32(d[22:26]),
        "stall_current_mA": i16(d[26:28]), "stall_protect": bool(d[28]), "key_lock": bool(d[29]),
        "vel_P": u32(d[30:34]), "vel_I": u32(d[34:38]), "vel_D": u32(d[38:42]),
        "auto_screen_off": bool(d[42]), "io_start_level": d[43],
    }


def decode_homing(d: bytes) -> dict:
    """0x94 read homing parameters (17 bytes)."""
    return {
        "auto_home_on_power": bool(d[0]), "home_state": d[1], "limit_current_mA": i16(d[2:4]),
        "left_origin": i32(d[4:8]), "timeout_ms": u32(d[8:12]), "right_origin": i32(d[12:16]),
        "limit_switches_on": bool(d[16]),
    }


# ======================================================================= client
@dataclass
class TxRecord:
    t: float
    tx: bytes
    rx: bytes
    latency: float | None
    error: str


class PD42S1:
    """Thread-safe request/response client for one driver on a pySerial-like port."""

    def __init__(self, port, protocol: str = "custom", address: int = 1, timeout: float = 0.03,
                 record: bool = True, log=None):
        if protocol not in ("custom", "modbus"):
            raise ValueError("protocol must be 'custom' or 'modbus'")
        self.port, self.protocol, self.address, self.timeout = port, protocol, address, timeout
        self.lock = threading.Lock()
        self.records: list[TxRecord] = []
        self.record = record
        self.log = log  # optional callable(str) for a GUI log panel
        self.n_ok = self.n_timeout = self.n_err = 0
        self.zero_speed_rpm: float | None = None  # learned: 0.0 if accepted, else 0.1

    # ------------------------------------------------------------ low level
    def _exchange(self, tx: bytes, parse) -> tuple[bytes, float]:
        with self.lock:
            if hasattr(self.port, "reset_input_buffer"):
                self.port.reset_input_buffer()
            t0 = time.perf_counter()
            self.port.write(tx)
            rx = bytearray()
            err = "timeout"
            while True:
                chunk = self.port.read(getattr(self.port, "in_waiting", 0) or 1)
                if chunk:
                    rx.extend(chunk)
                    data, err = parse(bytes(rx))
                    if data is not None:
                        lat = time.perf_counter() - t0
                        self._rec(t0, tx, bytes(rx), lat, "")
                        self.n_ok += 1
                        return data, lat
                    if err not in ("incomplete", "timeout"):
                        break
                if time.perf_counter() - t0 > self.timeout:
                    if err == "incomplete":
                        err = "short/garbled reply" if rx else "timeout"
                    break
        lat = time.perf_counter() - t0
        self._rec(t0, tx, bytes(rx), None, err)
        if err in ("timeout", "short/garbled reply"):
            self.n_timeout += 1
        else:
            self.n_err += 1
        raise DriverError(f"{hexs(tx)} -> {err} ({lat * 1e3:.1f} ms) rx={hexs(bytes(rx))}")

    def _custom(self, code: int, data: bytes = b"", n_reply: int = 0) -> bytes:
        tx = build_custom(self.address, code, data)

        def parse(rx: bytes):
            for fr in FrameParser().feed(rx):
                if fr.address == self.address and fr.func == code:
                    if fr.status != 0x01:
                        return None, f"driver error 0x{fr.status:02X} {CUSTOM_ERRORS.get(fr.status, '?')}"
                    return fr.body[1:], ""
            return None, "incomplete"

        d, _ = self._exchange(tx, parse)
        if len(d) < n_reply:
            raise DriverError(f"reply to 0x{code:02X} too short: {hexs(d)}")
        return d

    def _modbus(self, code: int, data: bytes | None) -> bytes:
        tx = build_modbus_read(self.address, code) if data is None else build_modbus_write(self.address, code, data)
        d, _ = self._exchange(tx, lambda rx: check_modbus_reply(tx, rx))
        if data is None:
            n = READ_LEN[code]
            d = d[1:] if len(d) == n + 1 else d  # odd length: drop the padding byte
            return d[:n] if len(d) >= n else d
        return d

    def read(self, code: int) -> bytes:
        if self.protocol == "custom":
            return self._custom(code, b"", READ_LEN[code])
        return self._modbus(code, None)

    def write(self, code: int, data: bytes = b"") -> bytes:
        if self.protocol == "custom":
            return self._custom(code, data)
        return self._modbus(code, data)

    def _rec(self, t0, tx, rx, lat, err):
        if self.record:
            self.records.append(TxRecord(t0, tx, rx, lat, err))
            if len(self.records) > 200000:
                del self.records[:100000]
        if self.log:
            self.log(f"[发] {hexs(tx)}\n[收] {hexs(rx)}" + (f"  !! {err}" if err else ""))

    # ------------------------------------------------------------ commands
    def enable(self):
        self.write(0xFA, b"\x00")          # 0 = enable (manual)

    def disable(self):
        self.write(0xFA, b"\x01")          # 1 = disable

    def clear_state(self):
        self.write(0xFB)                    # clears stall / brake / disabled state

    def brake(self):
        self.write(0xFC)                    # immediate stop, LATCHES: clear_state + enable to recover

    def recover(self):
        """Official sequence after a brake or stall: clear state, then enable."""
        self.clear_state()
        self.enable()

    def zero_position(self):
        self.write(0xF8)

    def release_stall(self):
        self.write(0xF9)

    def set_mode(self, mode: int):
        self.write(0x62, bytes([mode]))

    def set_echo(self, on: bool):
        if self.protocol == "custom":
            self.write(0x6F, bytes([0 if on else 1]))

    def set_limit_switches(self, on: bool):
        self.write(0x99, bytes([1 if on else 0]))

    def save_params(self):
        self.write(0x04)

    def set_speed(self, rpm: float, accel: int = 0) -> None:
        """Signed speed [rpm]. Zero is sent as 0.0; if the firmware rejects it
        (range 0.1..6000 in the manual) the client falls back to 0.1 rpm."""
        if abs(rpm) < 0.1:
            if self.zero_speed_rpm is None:
                try:
                    self.write(0xF1, speed_payload(0.0, accel))
                    self.zero_speed_rpm = 0.0
                    return
                except DriverError:
                    self.zero_speed_rpm = 0.1
            rpm = self.zero_speed_rpm if rpm >= 0 else -self.zero_speed_rpm
            if rpm == 0.0:
                self.write(0xF1, speed_payload(0.0, accel))
                return
        self.write(0xF1, speed_payload(rpm, accel))

    def read_position(self) -> int:
        return i32(self.read(0x2A))

    def read_speed(self) -> int:
        return i16(self.read(0x29))

    def read_status(self) -> int:
        return self.read(0x2C)[0]

    def read_stall(self) -> bool:
        return bool(self.read(0x2D)[0])

    def read_enabled(self) -> bool:
        return bool(self.read(0x2F)[0])     # 0x2F: 1 = enabled

    def read_bus_voltage(self) -> float:
        return f32(self.read(0x24))

    def read_version(self) -> tuple[float, float]:
        d = self.read(0x20)
        return d[0] / 10.0, d[1] / 10.0

    def read_system(self) -> dict:
        return decode_system(self.read(0x31))

    def read_drive(self) -> dict:
        return decode_drive(self.read(0x32))

    def read_homing(self) -> dict:
        return decode_homing(self.read(0x94))

    # ------------------------------------------------------------ readiness
    def readiness(self) -> tuple[bool, list[str], dict]:
        """Checks before closed-loop control. Returns (ok, problems, info)."""
        problems = []
        info = {}
        try:
            info["drive"] = drv = self.read_drive()
            info["system"] = sysp = self.read_system()
        except DriverError as e:
            return False, [f"driver not answering: {e}"], info
        if drv["work_mode"] != 1:
            problems.append(f"工作模式为 {drv['work_mode_name']}，需要 通信速度模式(1)")
        if self.protocol == "custom" and not drv["echo"]:
            problems.append("指令回响已关闭，本软件需要回响以确认每条指令")
        if sysp["bus_voltage_V"] < 11.0:
            problems.append(f"总线电压 {sysp['bus_voltage_V']:.2f} V < 11 V（12 V 电源开关是否打开？）")
        if sysp["stalled"]:
            problems.append("堵转标志置位：先“清除电机状态”再“使能”")
        try:
            if not self.read_enabled():
                problems.append("电机未使能")
        except DriverError as e:
            problems.append(f"读使能状态失败: {e}")
        return not problems, problems, info
