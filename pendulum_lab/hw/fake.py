"""Fake rig: pySerial-like ports backed by a real-time physics thread.

Used for software-in-the-loop tests of the *hardware* code path (sensor
parser, PD42S1 codec, motor thread, real-time loop, supervisor) without the
rig, and for classroom demos of the hardware GUI. The motor port implements
whatever set_speed / read_position layouts the protocol table defines.
"""
from __future__ import annotations

import math
import struct
import threading
import time

import numpy as np

from ..model.dynamics import rk4_pendulum
from ..model.params import PlantParams
from ..sim.simulator import DriverModel, SensorModel
from .pd42s1 import CUSTOM_WRITE_EXTRA, READ_LEN, FrameParser, _with_crc, build_custom, crc16_modbus


class FakeRig(threading.Thread):
    def __init__(self, p: PlantParams, protocol: str = "custom", address: int = 1, theta0: float = math.radians(2.0),
                 driver: DriverModel | None = None, sensor: SensorModel | None = None,
                 sensor_rate: float = 500.0, center_counts: float = 100000.0,
                 meters_per_rev: float = 0.040, rail_half: float = 0.30, reply_delay: float = 0.001, seed: int = 3,
                 mode: int = 1):
        super().__init__(daemon=True, name="fake-rig")
        self.p, self.protocol, self.address = p, protocol, address
        self.drv = driver or DriverModel(latency=0.0)
        self.sen = sensor or SensorModel(latency=0.0)
        self.sensor_rate = sensor_rate
        # 51200 counts/turn, meters_per_rev belt travel per turn, positive rpm -> +x
        self.cpm = 51200.0 / meters_per_rev
        self.center = center_counts
        self.spm = 60.0 / meters_per_rev  # rpm per (m/s)
        self.enabled, self.braked, self.mode, self.echo = True, False, mode, True
        self._mb = bytearray()
        self.rail_half = rail_half
        self.reply_delay = reply_delay
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()
        self.theta, self.omega = theta0, 0.0
        self.x, self.v, self.v_target = 0.0, 0.0, 0.0
        self.hold_rod = True     # a "hand" holds the rod until released
        self.running = True
        self.sensor_port = FakeSerial(self)
        self.motor_port = FakeSerial(self)
        self.motor_port.on_write = self._on_motor_bytes
        self._parser = FrameParser()
        self.crashed = False

    # ----------------------------------------------------------- physics
    def run(self) -> None:
        dt = 0.0005
        t_next_sample = time.perf_counter()
        t_last = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            n = int((now - t_last) / dt)
            if n == 0:
                time.sleep(0.0003)
                continue
            t_last += n * dt
            with self.lock:
                for _ in range(min(n, 50)):
                    self._step(dt)
            if now >= t_next_sample:
                t_next_sample += 1.0 / self.sensor_rate
                if now - t_next_sample > 0.1:
                    t_next_sample = now
                with self.lock:
                    adc = self.sen.adc_of(self.theta, self.rng)
                mv = int(adc * 3300 / 4096)
                self.sensor_port.feed(f"ADC={adc}, {mv} mV\r\n".encode())

    def _step(self, dt: float) -> None:
        tau = self.drv.tau
        dv = (self.v_target - self.v) * (1 - math.exp(-dt / tau)) if tau > 0 else self.v_target - self.v
        dv = max(-self.drv.accel_limit * dt, min(self.drv.accel_limit * dt, dv))
        a = dv / dt
        self.v += dv
        self.x += self.v * dt
        if abs(self.x) > self.rail_half:
            self.x = math.copysign(self.rail_half, self.x)
            self.v = 0.0
            self.crashed = True
        if self.hold_rod:
            self.omega = 0.0
        else:
            self.theta, self.omega = rk4_pendulum(self.p, self.theta, self.omega, a, dt)

    def release(self) -> None:
        self.hold_rod = False

    # ----------------------------------------------------------- motor
    def _on_motor_bytes(self, data: bytes) -> None:
        """Emulates the PD42S1 per the manual (custom C5..5C or Modbus-RTU)."""
        replies = []
        if self.protocol == "custom":
            for fr in self._parser.feed(data):
                if fr.address == self.address:
                    replies.append(self._reply_custom(fr.func, fr.body))
        else:
            self._mb.extend(data)
            while len(self._mb) >= 8:
                req = self._parse_modbus_request()
                if req is None:
                    break
                replies.append(req)
        for r in replies:
            if r is not None:
                threading.Timer(self.reply_delay, self.motor_port.feed, args=(r,)).start()

    def _parse_modbus_request(self):
        b = self._mb
        f = b[1]
        n = 8 if f in (0x04, 0x06) else (9 + b[6] if f == 0x10 and len(b) >= 7 else None)
        if n is None or len(b) < n:
            return None
        fr = bytes(b[:n])
        del b[:n]
        if crc16_modbus(fr[:-2]) != fr[-2] | (fr[-1] << 8) or fr[0] != self.address:
            self._mb.clear()
            return None
        code = fr[3]
        if f == 0x04:
            ok, data = self._execute(code, None)
            if not ok:
                return _with_crc(bytes([self.address, 0x84, 3]))
            d = (b"\x00" + data) if len(data) % 2 else data
            return _with_crc(bytes([self.address, 0x04, len(d)]) + d)
        payload = fr[4:6] if f == 0x06 else fr[7:-2]
        if f == 0x06 and code in (0xFA, 0x62, 0x99, 0x6F):
            payload = payload[1:]
        ok, _ = self._execute(code, payload)
        if not ok:
            return _with_crc(bytes([self.address, f | 0x80, 3]))
        return fr if f == 0x06 else _with_crc(fr[:6])

    def _reply_custom(self, code: int, body: bytes) -> bytes:
        read = code in READ_LEN
        ok, data = self._execute(code, None if read else body)
        if not ok:
            return build_custom(self.address, code, b"\xE6")
        if not read:
            data = body[:CUSTOM_WRITE_EXTRA.get(code, 0)]
        return build_custom(self.address, code, b"\x01" + data)

    def _execute(self, code: int, payload):
        """Returns (ok, read_data)."""
        with self.lock:
            if payload is None:  # read
                cnt = int(round(self.center + self.x * self.cpm))
                rpm = int(round(self.v * self.spm))
                if code == 0x2A:
                    return True, struct.pack(">i", cnt)
                if code == 0x29:
                    return True, struct.pack(">h", rpm)
                if code == 0x2C:
                    return True, bytes([2 if abs(self.v) > 1e-4 else 0])
                if code == 0x2D:
                    return True, b"\x00"
                if code == 0x2F:
                    return True, bytes([1 if self.enabled else 0])
                if code == 0x24:
                    return True, struct.pack(">f", 12.1)
                if code == 0x20:
                    return True, bytes([10, 10])
                if code == 0x31:
                    d = struct.pack(">fhfff", 12.1, 0, 3.827, 1.73, 4.082) + struct.pack(">h", rpm)
                    d += struct.pack(">iiiI", cnt, cnt, 0, 0) + bytes([0 if self.enabled else 1, 1, 0])
                    return True, d
                if code == 0x32:
                    d = bytes([self.mode, 0 if self.echo else 1]) + struct.pack(">IH", 115200, 500) + bytes([0, 1])
                    d += struct.pack(">Hh", 16, 1000) + struct.pack(">III", 550, 8, 300) + struct.pack(">h", 2500)
                    d += bytes([1, 0]) + struct.pack(">III", 6, 45, 20) + bytes([0, 0])
                    return True, d
                if code == 0x94:
                    return True, bytes([0, 0]) + struct.pack(">hiIi", 300, 0, 10000, 0) + b"\x00"
                if code in READ_LEN:
                    return True, bytes(READ_LEN[code])
                return False, b""
            if code == 0xF1:
                if len(payload) != 6:
                    return False, b""
                direction = payload[0]  # payload[1] = accel byte (ignored: the twin uses DriverModel)
                rpm = struct.unpack(">f", payload[2:6])[0]
                if self.enabled and not self.braked and self.mode == 1:
                    self.v_target = (-rpm if direction else rpm) / self.spm
                return True, b""
            if code == 0xFA:
                self.enabled = payload[-1] == 0
                if not self.enabled:
                    self.v_target = 0.0
                return True, b""
            if code == 0xFC:
                self.braked = True
                self.v_target = 0.0
                self.v = 0.0
                return True, b""
            if code == 0xFB:
                self.braked = False
                return True, b""
            if code == 0x62:
                self.mode = payload[-1]
                return True, b""
            if code == 0x6F:
                self.echo = payload[-1] == 0
                return True, b""
            if code == 0xF8:
                self.center -= int(round(self.center + self.x * self.cpm))
                return True, b""
            return True, b""

    def stop(self) -> None:
        self.running = False


class FakeSerial:
    """Minimal pySerial look-alike (read/write/in_waiting/reset_input_buffer)."""

    def __init__(self, rig: FakeRig, timeout: float = 0.01):
        self.rig = rig
        self.timeout = timeout
        self.buf = bytearray()
        self.cv = threading.Condition()
        self.on_write = None
        self.is_open = True

    def feed(self, data: bytes) -> None:
        with self.cv:
            self.buf.extend(data)
            self.cv.notify_all()

    @property
    def in_waiting(self) -> int:
        with self.cv:
            return len(self.buf)

    def read(self, n: int = 1) -> bytes:
        with self.cv:
            if not self.buf:
                self.cv.wait(self.timeout)
            out = bytes(self.buf[:n])
            del self.buf[:n]
            return out

    def write(self, data: bytes) -> int:
        if self.on_write:
            self.on_write(bytes(data))
        return len(data)

    def reset_input_buffer(self) -> None:
        with self.cv:
            self.buf.clear()

    def close(self) -> None:
        self.is_open = False
