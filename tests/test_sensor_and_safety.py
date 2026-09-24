import math

import pytest

from pendulum_lab.control.safety import SafetyLimits, Supervisor
from pendulum_lab.hw.sensor import AngleCalibration, parse_line


@pytest.mark.parametrize("line,adc,t", [
    (b"ADC=1966, 1584 mV\r", 1966, None),
    (b"ADC=4093, 3298 mV", 4093, None),
    (b"ADC=2001,T=123456", 2001, 123456),
    (b"1999", 1999, None),
    (b"hello", None, None),
])
def test_parse_line(line, adc, t):
    assert parse_line(line) == (adc, t)


def test_calibration_from_rig_readings():
    cal = AngleCalibration.from_upright_and_hanging(1966, 4092)
    assert cal.counts_per_rad == pytest.approx(676.7, abs=0.1)
    assert math.degrees(cal.theta(4092)) == pytest.approx(180.0)
    assert cal.deg_per_count == pytest.approx(0.0847, abs=1e-3)


def test_supervisor_trips_on_predicted_stop_before_the_limit():
    sup = Supervisor(SafetyLimits(x_soft=0.2, a_brake=3.0, latency=0.03))
    assert sup.check(0.0, x=0.10, v=0.0, theta=0.0)
    sup2 = Supervisor(SafetyLimits(x_soft=0.2, a_brake=3.0, latency=0.03))
    assert not sup2.check(0.0, x=0.10, v=0.8, theta=0.0)  # would stop at ~0.23 m
    assert "predicted stop" in sup2.reason


def test_stop_command_ramps_to_zero_without_overshoot():
    v, Ts = 0.37, 0.005
    for _ in range(200):
        v += Supervisor.stop_command(v, 3.0, Ts) * Ts
    assert v == 0.0
