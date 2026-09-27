"""PD42S1 codec checked byte-for-byte against the official upper computer's logs
(custom protocol screenshot and Modbus-RTU screenshot) and the V1.2 manuals."""
import struct

import pytest

from pendulum_lab.hw.pd42s1 import (
    PD42S1, FrameParser, build_custom, build_modbus_read, build_modbus_write, check_modbus_reply,
    crc16_modbus, decode_homing, decode_system, parse_hex, speed_payload, sum8, unpad_read,
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


# --- Modbus-RTU log (设备系统 page, 2026-09-27): read version and system parameters
MODBUS_READS = [
    ("01 04 00 20 00 01 30 00", "01 04 02 11 0A 35 67"),
    ("01 04 00 31 00 14 A1 CA",
     "01 04 28 41 C3 4F F4 FF F8 40 74 F3 AA 3F DD 76 5E 40 82 9E BC 00 00 00 00 96 B8 00 00 96 CA "
     "FF FF FF EE 00 00 00 00 00 00 00 00 CE A6"),
]


def test_modbus_read_frames_from_the_official_log():
    for tx, rx in MODBUS_READS:
        t, r = parse_hex(tx), parse_hex(rx)
        assert build_modbus_read(t[0], t[3]) == t
        payload, err = check_modbus_reply(t, r)
        assert err == "" and len(payload) == r[2]  # byte count, CRC low byte first


def test_odd_length_read_pad_is_trailing():
    """The 39-byte system block comes back as 40 bytes with the 0x00 pad LAST;
    decoding must reproduce exactly the values the official software shows."""
    t, r = parse_hex(MODBUS_READS[1][0]), parse_hex(MODBUS_READS[1][1])
    payload, _ = check_modbus_reply(t, r)
    s = decode_system(unpad_read(payload, 39))
    assert s["bus_voltage_V"] == pytest.approx(24.41, abs=5e-3)
    assert s["phase_current_mA"] == -8
    assert s["flux_mWb"] == pytest.approx(3.827, abs=1e-3)
    assert s["phase_R_ohm"] == pytest.approx(1.730, abs=1e-3)
    assert s["phase_L_mH"] == pytest.approx(4.082, abs=1e-3)
    assert (s["speed_rpm"], s["target_pos"], s["position"], s["pos_error"], s["pulse_count"]) == (0, 38584, 38602, -18, 0)
    assert s["enabled"] and not s["in_position"] and not s["stalled"]   # 电机使能 / 未到位 / 未堵转


def test_one_byte_read_accepts_either_pad_position():
    assert unpad_read(b"\x01\x00", 1) == b"\x01"
    assert unpad_read(b"\x00\x01", 1) == b"\x01"


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


def test_driver_selftest_rehearsal(tmp_path):
    """Guided self-test (steps 0-2) against the emulator: torque mode -> speed mode, 0 rpm stops."""
    import math

    from pendulum_lab.config import load_config, plant_from
    from pendulum_lab.hw.fake import FakeRig
    from pendulum_lab.tools.drivertest import DriverSelfTest

    rig = FakeRig(plant_from(load_config()), "modbus", theta0=math.pi, mode=2)
    rig.start()
    answers = iter(["y", "y", "n", "", "R", "y"])  # values ok, switch mode, don't save, start, + goes right, disable
    try:
        t = DriverSelfTest(PD42S1(rig.motor_port, "modbus", timeout=0.05), out=lambda m: None,
                           ask=lambda p: next(answers), log_dir=str(tmp_path))
        res = t.run(("0", "1", "2"))
    finally:
        rig.stop()
    assert rig.mode == 1 and not rig.enabled                    # speed mode, left disabled at the end
    assert res["Q2"]["decode_plausible"] and res["Q2"]["one_byte_read_consistent"]
    assert all(v["accepted"] and v["ok"] for v in res["Q1"].values())
    assert t.plus_side == "右"
    assert t.log_path.exists() and t.frames_path.exists()
    assert "01 06 00 62 00 01" in t.frames_path.read_text()     # set work mode 1 (06H, value in the low byte)


def test_enable_state_semantics_from_the_rig_log():
    """Self-test log 2026-09-27: motor enabled (official GUI: 电机使能), 0x31 byte 37 = 00 and
    0x2F answered 00. The client used to read 0x2F as "1 = enabled" -> "电机未使能" always."""
    sys_req = parse_hex("01 04 00 31 00 14 A1 CA")
    sys_rx = parse_hex("01 04 28 41 C3 3A 80 00 00 40 74 F3 AA 3F DD 76 5E 40 82 9E BC 00 00 00 00 9E BB "
                       "00 00 9E BB 00 00 00 00 00 00 00 00 00 00 00 00 E9 2D")
    payload, err = check_modbus_reply(sys_req, sys_rx)
    s = decode_system(unpad_read(payload, 39))
    assert err == "" and s["enabled"] and s["bus_voltage_V"] == pytest.approx(24.4, abs=0.05)
    en_req, en_rx = parse_hex("01 04 00 2F 00 01 00 03"), parse_hex("01 04 02 00 00 B9 30")
    payload, err = check_modbus_reply(en_req, en_rx)
    assert err == "" and unpad_read(payload, 1) == b"\x00"   # 0 while enabled


@pytest.mark.parametrize("protocol", ["custom", "modbus"])
def test_closed_loop_preflight_enables_a_disabled_driver(protocol):
    """After drivertest (or a brake) the motor is disabled: preflight must run the official
    clear -> enable -> zero-speed sequence itself and centre the soft limits at the start."""
    import math

    from pendulum_lab.config import load_config, plant_from
    from pendulum_lab.hw.fake import FakeRig
    from pendulum_lab.hw.motor import fake_units, make_worker
    from pendulum_lab.hw.runtime import HardwareLoop, RunOptions
    from pendulum_lab.hw.sensor import AngleCalibration, SensorReader

    cfg = load_config()
    rig = FakeRig(plant_from(cfg), protocol, theta0=math.radians(1.0))
    rig.enabled, rig.braked = False, True
    rig.x = 0.07                                   # cart put 7 cm off the old centre
    rig.start()
    sensor = SensorReader(rig.sensor_port)
    sensor.start()
    drv = PD42S1(rig.motor_port, protocol, timeout=0.05)
    assert not drv.readiness()[0]
    motor = make_worker(drv, cfg["hardware"], fake_units())
    motor.start()
    try:
        loop = HardwareLoop(cfg, sensor, motor, RunOptions("lqr", "closed", 1.0), AngleCalibration(), printer=lambda m: None)
        info = loop.preflight()
        assert rig.enabled and not rig.braked
        assert info["x0"] == pytest.approx(0.0, abs=1e-3)   # centre = start position
        assert loop.sup.lim.x_soft == pytest.approx(0.25)
    finally:
        motor.stop()
        sensor.stop()
        rig.stop()
