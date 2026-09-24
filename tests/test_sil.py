"""Software-in-the-loop: the real hardware code path against the fake rig (real time, ~5 s)."""
import math
import threading
import time

import pytest

from pendulum_lab.config import load_config, plant_from
from pendulum_lab.hw.fake import FakeRig
from pendulum_lab.hw.motor import fake_units, make_worker
from pendulum_lab.hw.pd42s1 import PD42S1
from pendulum_lab.hw.runtime import HardwareLoop, RunOptions
from pendulum_lab.hw.sensor import AngleCalibration, SensorReader


@pytest.mark.parametrize("protocol", ["custom", "modbus"])
def test_hardware_loop_balances_fake_rig(protocol, tmp_path):
    algo = "lqr"
    cfg = load_config()
    rig = FakeRig(plant_from(cfg), protocol, theta0=math.radians(1.0))
    rig.start()
    sensor = SensorReader(rig.sensor_port)
    sensor.start()
    motor = make_worker(PD42S1(rig.motor_port, protocol, timeout=0.02), cfg["hardware"], fake_units())
    motor.start()
    log = tmp_path / "run.csv"
    loop = HardwareLoop(cfg, sensor, motor, RunOptions(algo, "closed", 3.0, log_path=str(log)), AngleCalibration(),
                        printer=lambda m: None)
    th = threading.Thread(target=loop.run)
    th.start()
    t0 = time.time()
    while loop.live.state != "ACTIVE" and time.time() - t0 < 10:
        time.sleep(0.01)
    rig.release()
    th.join(timeout=20)
    try:
        assert loop.result_message == "duration reached"
        assert not rig.crashed
        assert abs(math.degrees(rig.theta)) < 2.0
        assert motor.total_errors == 0
        assert log.exists()
    finally:
        motor.stop()
        sensor.stop()
        rig.stop()


def test_shadow_mode_never_writes_to_motor():
    cfg = load_config()
    rig = FakeRig(plant_from(cfg), "custom", theta0=math.radians(0.5))
    writes = []
    rig.motor_port.on_write = writes.append
    rig.start()
    sensor = SensorReader(rig.sensor_port)
    sensor.start()
    loop = HardwareLoop(cfg, sensor, None, RunOptions("lqr", "shadow", 1.0), AngleCalibration(), printer=lambda m: None)
    loop.run()
    sensor.stop()
    rig.stop()
    assert loop.result_message == "duration reached"
    assert writes == []
    # rod held leaning +0.5 deg toward +x -> the controller would accelerate toward +x
    a_ctrl = [r[13] for r in loop.rows if r[2] == "ACTIVE"]
    assert a_ctrl and sum(a_ctrl) / len(a_ctrl) > 0
