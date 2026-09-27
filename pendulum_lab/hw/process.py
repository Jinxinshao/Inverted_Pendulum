"""Run the hardware loop in its own OS process (used by the GUI).

Why: CPython threads share one interpreter lock (GIL). Matplotlib's Agg
renderer holds it for tens of milliseconds per redraw; measured on the fake
rig, a control THREAD inside the GUI process saw loop gaps up to 54 ms
(p99 9.5 ms) - as large as the loop's delay margin. A separate PROCESS has
its own interpreter, so the GUI can never stall the control loop.

Communication: the child pushes batches of log rows and state messages into
a multiprocessing.Queue; the GUI sets a multiprocessing.Event to stop.
Uses the 'spawn' start method on every OS (Windows only has spawn).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import threading
import time
from pathlib import Path


def _build_and_run(spec: dict, q, stop_ev) -> None:  # runs in the child process
    from ..config import plant_from
    from .motor import fake_units, make_worker, open_driver
    from .pd42s1 import PD42S1
    from .runtime import HardwareLoop, RunOptions, calibration_from
    from .sensor import AngleCalibration, SensorReader

    cfg, mode = spec["cfg"], spec["mode"]
    hw = cfg["hardware"]
    objs: list = []
    rig = None
    try:
        if mode == "sil":
            from .fake import FakeRig

            rig = FakeRig(plant_from(cfg), hw.get("motor_protocol", "custom"), theta0=math.radians(1.0))
            rig.start()
            sensor = SensorReader(rig.sensor_port)
            drv = PD42S1(rig.motor_port, hw.get("motor_protocol", "custom"), timeout=0.05)  # same as open_driver
            motor = make_worker(drv, hw, fake_units())
            calib = AngleCalibration()
            run_mode = "closed"
            objs = [rig]
        else:
            from ..tools.hwtools import open_serial

            calib = calibration_from(hw)
            motor = None
            if mode == "closed":
                drv, mport = open_driver(hw, spec["motor_port"])
                objs.append(mport)
                motor = make_worker(drv, hw)
            sport = open_serial(spec["sensor_port"], hw["sensor_baud"])
            objs.append(sport)
            sensor = SensorReader(sport)
            run_mode = mode
        sensor.start()
        objs.insert(0, sensor)
        if motor is not None:
            motor.start()
            objs.insert(0, motor)
        loop = HardwareLoop(cfg, sensor, motor, RunOptions(spec["algo"], run_mode, spec["duration"], log_path=spec["log"],
                                                            adapt=spec.get("adapt", "monitor"),
                                                            excitation_amp=spec.get("excitation", 0.0)),
                            calib, printer=lambda m: q.put(("msg", m)))

        def forward():
            sent = 0
            while True:
                if stop_ev.is_set():
                    loop.request_stop()
                with loop.live.lock:
                    st, msg = loop.live.state, loop.live.message
                n = len(loop.rows)
                if n > sent:
                    q.put(("rows", loop.rows[sent:n]))
                    sent = n
                q.put(("state", st, msg))
                if rig is not None and st in ("ARMED", "ACTIVE") and rig.hold_rod:
                    rig.release()
                if st == "DONE":
                    return
                time.sleep(0.05)

        fw = threading.Thread(target=forward, daemon=True)
        fw.start()
        result = loop.run()
        fw.join(timeout=1.0)
        q.put(("done", result, getattr(loop, "overruns", 0), spec["log"]))
    except Exception as e:  # noqa: BLE001
        q.put(("done", f"ERROR: {type(e).__name__}: {e}", 0, None))
    finally:
        for o in objs:
            for m in ("stop", "close"):
                f = getattr(o, m, None)
                if f:
                    try:
                        f()
                    except Exception:  # noqa: BLE001
                        pass


class LoopProcess:
    def __init__(self, cfg: dict, mode: str, algo: str, duration: float, sensor_port: str = "", motor_port: str = "",
                 log: str | None = None, adapt: str = "monitor", excitation: float = 0.0):
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue()
        self.stop_ev = ctx.Event()
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
        spec = dict(cfg=cfg, mode=mode, algo=algo, duration=duration, sensor_port=sensor_port, motor_port=motor_port, log=log,
                    adapt=adapt, excitation=excitation)
        self.proc = ctx.Process(target=_build_and_run, args=(spec, self.q, self.stop_ev), daemon=True)
        self.rows: list = []
        self.state, self.message = "STARTING", ""
        self.done = None

    def start(self) -> None:
        self.proc.start()

    def stop(self) -> None:
        self.stop_ev.set()

    def poll(self) -> None:
        while True:
            try:
                item = self.q.get_nowait()
            except Exception:  # noqa: BLE001  (queue.Empty)
                break
            if item[0] == "rows":
                self.rows.extend(item[1])
                if len(self.rows) > 6000:
                    del self.rows[:-4000]
            elif item[0] == "state":
                self.state, self.message = item[1], item[2]
            elif item[0] == "msg":
                self.message = item[1]
            elif item[0] == "done":
                self.done = item[1:]
                self.state = "DONE"
                self.message = item[1]

    @property
    def alive(self) -> bool:
        return self.proc.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.terminate()


def wait_until_done(lp: LoopProcess, timeout: float = 60.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout and lp.done is None:
        lp.poll()
        time.sleep(0.05)
