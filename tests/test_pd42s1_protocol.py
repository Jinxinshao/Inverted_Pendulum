"""PD42S1 codec checked byte-for-byte against the official upper computer's logs
(custom protocol screenshot and Modbus-RTU screenshot) and the V1.2 manuals."""
import struct

import pytest

from pendulum_lab.hw.pd42s1 import (
    PD42S1, FrameParser, build_custom, build_modbus_read, build_modbus_write, check_modbus_reply,
    crc16_modbus, decode_homing, parse_hex, speed_payload, sum8,
)

# --- custom protocol log (设备系统 page) -------------------------------------
CUSTOM = [
    ("C5 01 FC C2 5C", "C5 01 FC 01 C3 5C"),                  # 立即停止(刹车)
    ("C5 01 FA 00 C0 5C", "C5 01 FA 01 00 C1 5C"),            # 使能 (00 = enable)
    ("C5 01 94 5A 5C", "C5 01 94 01 00 00 01 2C 00 00 00 00 00 00 27 10 00 00 00 00 00 BF 5C"),  # 读回零参数
    ("C5 01 F4 BA 5C", "C5 01 F4 01 BB 5C"),                  # 脉冲模式控制
    ("C5 01 F2 00 65 00 08 00 00 00 00 25 5C", "C5 01 F2 01 B9 5C"),  # 绝对位置模式
]
# --- Modbus-RTU log (运动控制 page) -------------------------------------------
MODBUS = [
    ("01 06 00 FA 00 00 A9 FB", "01 06 00 FA 00 00 A9 FB"),   # 使能
    ("01 06 00 FA 00 01 68 3B", "01 06 00 FA 00 01 68 3B"),   # 失能
    ("01 10 00 F1 00 03 06 00 0A 42 48 00 00 BE E9", "01 10 00 F1 00 03 D1 FB"),  # 速度模式 正转 acc10 50rpm
    ("01 10 00 F1 00 03 06 01 0A 42 48 00 00 BF 38", "01 10 00 F1 00 03 D1 FB"),  # 反转
]


@pytest.mark.parametrize("tx,rx", CUSTOM)
def test_custom_frames_rebuild_exactly(tx, rx):
    t, r = parse_hex(tx), parse_hex(rx)
    assert build_custom(t[1], t[2], t[3:-2]) == t
    assert sum8(r[:-2]) == r[-2] and r[3] == 0x01  # reply: err byte 01 = OK


def test_custom_log_semantics():
    # F2 absolute position: dir 0, accel 101 (0x65), speed 8 rpm, position 0 -> matches the UI
    body = parse_hex(CUSTOM[4][0])[3:-2]
    assert body[0] == 0 and body[1] == 101 and struct.unpack(">H", body[2:4])[0] == 8
    # 0x94 homing: limit current 300 mA, timeout 10000 ms, limits off -> matches the UI
    h = decode_homing(parse_hex(CUSTOM[2][1])[4:-2])
    assert h["limit_current_mA"] == 300 and h["timeout_ms"] == 10000 and not h["limit_switches_on"]


@pytest.mark.parametrize("tx,rx", MODBUS)
def test_modbus_frames_rebuild_exactly(tx, rx):
    t, r = parse_hex(tx), parse_hex(rx)
    if t[1] == 0x06:
        built = build_modbus_write(t[0], t[3], t[4:6])
    else:
        built = build_modbus_write(t[0], t[3], t[7:-2])
    assert built == t
    assert crc16_modbus(r[:-2]) == r[-2] | (r[-1] << 8)  # CRC low byte first on the wire
    payload, err = check_modbus_reply(t, r)
    assert err == "" and payload is not None


def test_speed_payload_matches_ui():
    assert speed_payload(50.0, 10) == parse_hex("00 0A 42 48 00 00")
    assert speed_payload(-50.0, 10) == parse_hex("01 0A 42 48 00 00")


def test_modbus_read_request_and_padding():
    assert build_modbus_read(1, 0x2A)[:6] == bytes([1, 4, 0, 0x2A, 0, 2])
    assert build_modbus_read(1, 0x2C)[:6] == bytes([1, 4, 0, 0x2C, 0, 1])   # 1 byte -> 1 register
    assert build_modbus_read(1, 0x31)[:6] == bytes([1, 4, 0, 0x31, 0, 20])  # 39 bytes -> 20 registers


def test_modbus_exception_is_reported():
    req = build_modbus_write(1, 0xF1, speed_payload(10, 0))
    exc = bytes([1, 0x90, 3])
    c = crc16_modbus(exc)
    payload, err = check_modbus_reply(req, exc + bytes([c & 0xFF, c >> 8]))
    assert payload is None and "exception 3" in err


def test_parser_handles_garbage_split_and_concatenated_frames():
    p = FrameParser()
    stream = b"\x00\x13garbage" + parse_hex(CUSTOM[2][1]) + parse_hex(CUSTOM[0][1])
    frames = []
    for i in range(0, len(stream), 3):
        frames += p.feed(stream[i:i + 3])
    assert [f.func for f in frames] == [0x94, 0xFC]


def test_parser_skips_tail_byte_inside_payload():
    f = build_custom(1, 0x2A, bytes([1, 0x00, 0x5C, 0x12, 0x34]))
    assert FrameParser().feed(f)[0].body == bytes([1, 0x00, 0x5C, 0x12, 0x34])


@pytest.mark.parametrize("protocol", ["custom", "modbus"])
def test_driver_against_emulator(protocol):
    """High-level API against the manual-based emulator (both protocols)."""
    import math

    from pendulum_lab.hw.fake import FakeRig
    from pendulum_lab.model.params import default_params

    rig = FakeRig(default_params(), protocol, theta0=math.radians(1.0), mode=2)
    rig.start()
    try:
        d = PD42S1(rig.motor_port, protocol, timeout=0.05)
        ok, problems, info = d.readiness()
        assert not ok and any("通信速度模式" in p for p in problems)  # starts in torque mode
        d.set_mode(1)
        d.brake()
        d.set_speed(60.0, 0)
        assert abs(rig.v_target) < 1e-9        # braked: ignores speed commands
        d.recover()                            # clear state + enable (official sequence)
        assert d.readiness()[0]
        d.set_speed(60.0, 0)                   # 60 rpm * 40 mm = 0.04 m/s
        assert rig.v_target == pytest.approx(0.04, rel=1e-6)
        d.set_speed(-30.0, 0)
        assert rig.v_target == pytest.approx(-0.02, rel=1e-6)
        pos = d.read_position()
        assert isinstance(pos, int)
        sysp = d.read_system()
        assert sysp["bus_voltage_V"] == pytest.approx(12.1, rel=1e-5)
        d.set_speed(0.0, 0)
        assert d.zero_speed_rpm == 0.0
    finally:
        rig.stop()
