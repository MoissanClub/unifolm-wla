"""Explicitly enabled, free-space arm test. Built-in locomotion retains balance.

Not a contact controller. No hand commands, leg commands, or autonomous walking.
The remote and the firmware controller remain the operator's emergency stop.
"""
from __future__ import annotations

import threading
import time
import numpy as np

from .robot import ARM_IDS


class ArmSession:
    def __init__(self, sensors):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        self.sensors = sensors
        self.loco = LocoClient()
        self.loco.SetTimeout(0.2)
        self.loco.Init()
        code, fsm = self.loco.GetFsmId()
        if code != 0 or fsm != 500:
            raise RuntimeError(f"Expected standing main controller FSM 500, got {code}, {fsm}")
        low, q, _ = sensors.feedback()
        self.command = {s: q[list(ids)].copy() for s, ids in ARM_IDS.items()}
        self.target = {s: values.copy() for s, values in self.command.items()}
        self.waist = q[12]
        self.weight = 0.0
        self.fault = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.message = unitree_hg_msg_dds__LowCmd_()
        self.message.mode_pr = 0
        self.message.mode_machine = low.mode_machine
        self.crc = CRC()
        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        try:
            for weight in np.linspace(0, 1, 200):
                self.check()
                with self.lock:
                    self.weight = float(weight)
                time.sleep(0.01)
        except BaseException:
            self.close()
            raise

    def _loop(self):
        deadline, last_fsm = time.monotonic(), 0.0
        while not self.stop_event.is_set():
            try:
                low, measured, _ = self.sensors.feedback()
                if abs(low.imu_state.rpy[0]) > 0.2 or abs(low.imu_state.rpy[1]) > 0.2:
                    raise RuntimeError("Excessive body tilt")
                if time.monotonic()-last_fsm > 0.5:
                    code, fsm = self.loco.GetFsmId()
                    last_fsm = time.monotonic()
                    if code != 0 or fsm != 500:
                        raise RuntimeError("Main controller FSM changed")
                with self.lock:
                    for side, ids in ARM_IDS.items():
                        if np.max(np.abs(measured[list(ids)] - self.command[side])) > 0.15:
                            raise RuntimeError(f"{side} tracking error")
                        if self.fault is None:
                            self.command[side] += np.clip(self.target[side]-self.command[side], -0.005, 0.005)
            except Exception as exc:
                with self.lock:
                    self.fault = str(exc)
                    self.target = {s: values.copy() for s, values in self.command.items()}
            try:
                with self.lock:
                    for side, ids in ARM_IDS.items():
                        for j, motor in enumerate(ids):
                            cmd = self.message.motor_cmd[motor]
                            cmd.q, cmd.dq, cmd.tau = float(self.command[side][j]), 0.0, 0.0
                            cmd.kp, cmd.kd = (40.0, 1.5) if j == 4 else (60.0, 1.5)
                    waist = self.message.motor_cmd[12]
                    waist.q, waist.dq, waist.tau, waist.kp, waist.kd = float(self.waist), 0.0, 0.0, 60.0, 1.5
                    self.message.motor_cmd[29].q = self.weight
                    self.message.crc = self.crc.Crc(self.message)
                    self.publisher.Write(self.message)
            except Exception as exc:
                self.fault = f"DDS write failed: {exc}"
                break
            deadline += 0.01
            time.sleep(max(0, deadline-time.monotonic()))
            if time.monotonic()-deadline > 0.05:
                deadline = time.monotonic()

    def check(self):
        if self.fault is not None or not self.thread.is_alive():
            raise RuntimeError(self.fault or "Arm publisher stopped")

    def move(self, targets, duration):
        self.check()
        with self.lock:
            self.target = {s: q.copy() for s, q in targets.items()}
        end = time.monotonic()+duration
        while time.monotonic() < end:
            self.check()
            time.sleep(0.01)

    def close(self):
        # Free-space test only: smoothly hand authority back to the standing controller.
        with self.lock:
            self.target = {s: values.copy() for s, values in self.command.items()}
            initial = self.weight
        for weight in np.linspace(initial, 0, 100):
            with self.lock:
                self.weight = float(weight)
            time.sleep(0.01)
        self.stop_event.set()
        self.thread.join(timeout=1)
        self.publisher.Close()
