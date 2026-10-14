"""Explicit arm_sdk ownership and one-arm/BrainCo trajectory execution.

The standing controller owns balance. The remote and firmware remain the
operator's emergency stop; this module never changes the locomotion FSM.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .hands import BrainCoHands
from .contract import FPS
from .robot import ARM_IDS


ARM_SERVICE = "g1_arm_example"
STREAM_HZ = 50.0  # Unitree's arm5 SDK example and the measured PC2 deployment.
HAND_HEARTBEAT_HZ = 10.0  # Keep the selected hand's position target active during inference.
DISCOVERY_S = 1.0  # DDS discovery settling measured in the fridge deployment.
RAMP_IN_S = 2.0
RAMP_OUT_S = 1.0


def interpolate(points, elapsed, fps):
    """Interpolate from points[0] through each model frame, then hold the end."""
    phase = max(0.0, min(float(elapsed) * fps, len(points) - 1))
    index = int(phase)
    if index == len(points) - 1:
        return points[-1].copy()
    fraction = phase - index
    return (1 - fraction) * points[index] + fraction * points[index + 1]


class ArmSession:
    def __init__(self, sensors, side="left", fps=FPS):
        if side not in ARM_IDS:
            raise ValueError("Arm side must be left or right")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Model fps must be positive and finite")
        self.sensors, self.side, self.fps = sensors, side, float(fps)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.publisher = self.hands = self.thread = self.services = None
        self.command = {}
        self.trajectory = None
        self.hand_target = None
        self.hand_sent_at = None
        self.weight = 0.0
        self.fault = None
        self.restore_service = False
        self.closed = False

    def _standing(self):
        code, fsm = self.loco.GetFsmId()
        if code != 0 or fsm != 500:
            raise RuntimeError(f"Expected standing main controller FSM 500, got {code}, {fsm}")

    def _service_status(self):
        code, services = self.services.ServiceList()
        if code != 0:
            raise RuntimeError(f"Cannot query arm-service ownership: {code}")
        for service in services:
            if service.name == ARM_SERVICE:
                # Unitree G1 RobotStateClient documentation: 0=running, 1=stopped.
                if service.status not in (0, 1):
                    raise RuntimeError(f"Unknown {ARM_SERVICE} status: {service.status}")
                return service.status
        raise RuntimeError(f"ServiceList does not contain {ARM_SERVICE}")

    def start(self):
        """Take arm authority only after the caller completes its preflight."""
        if self.closed or self.thread is not None:
            raise RuntimeError("Arm session cannot be started twice or after close")
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        # Same robot_state API used by G1; this import also works in the PC2 SDK.
        from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        self.loco = LocoClient()
        self.loco.Init()
        self.services = RobotStateClient()
        self.services.Init()
        # Keep SDK transport timeouts; don't replace them with an action deadline.
        time.sleep(DISCOVERY_S)
        self._standing()
        status = self._service_status()
        try:
            if status == 0:
                self.restore_service = True
                code = self.services.ServiceSwitch(ARM_SERVICE, False)
                if code != 0 or self._service_status() != 1:
                    raise RuntimeError(f"Cannot release built-in arm-service ownership: {code}")
            low, motors, _ = self.sensors.feedback()
            self.command = {side: motors[list(ids)].copy() for side, ids in ARM_IDS.items()}
            self.waist = float(motors[12])
            self.message = unitree_hg_msg_dds__LowCmd_()
            self.message.mode_pr = 0
            self.message.mode_machine = low.mode_machine
            self.crc = CRC()
            self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self.publisher.Init()
            self.hands = BrainCoHands()
            self.thread = threading.Thread(target=self._loop, name="wla-arm-sdk", daemon=True)
            self.thread.start()
            self._ramp(1.0, RAMP_IN_S)
        except BaseException:
            self.close()
            raise

    def _publish(self):
        with self.lock:
            if self.trajectory is not None:
                points, started, fps = self.trajectory
                self.command[self.side] = interpolate(points, time.monotonic() - started, fps)
            for side, ids in ARM_IDS.items():
                for index, motor in enumerate(ids):
                    command = self.message.motor_cmd[motor]
                    command.q = float(self.command[side][index])
                    command.dq, command.tau = 0.0, 0.0
                    command.kp, command.kd = (40.0, 1.5) if index == 4 else (60.0, 1.5)
            waist = self.message.motor_cmd[12]
            waist.q, waist.dq, waist.tau, waist.kp, waist.kd = self.waist, 0.0, 0.0, 60.0, 1.5
            self.message.motor_cmd[29].q = self.weight
            self.message.crc = self.crc.Crc(self.message)
            if self.publisher.Write(self.message) is False:
                raise RuntimeError("Arm SDK DDS write failed")
            if (not self.closed and self.hand_target is not None
                    and time.monotonic() - self.hand_sent_at >= 1 / HAND_HEARTBEAT_HZ):
                self.hands.write(self.side, self.hand_target)
                self.hand_sent_at = time.monotonic()

    def _loop(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                self._publish()
            except Exception as exc:
                self.fault = str(exc)
                self.stop_event.set()
                break
            self.stop_event.wait(max(0.0, 1 / STREAM_HZ - (time.monotonic() - started)))

    def check(self):
        if self.fault is not None:
            raise RuntimeError(self.fault)
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError("Arm publisher is not running")

    def _ramp(self, target, duration):
        initial, started = self.weight, time.monotonic()
        while duration > 0 and time.monotonic() - started < duration:
            self.check()
            fraction = (time.monotonic() - started) / duration
            with self.lock:
                self.weight = initial + min(fraction, 1.0) * (target - initial)
            time.sleep(1 / STREAM_HZ)
        with self.lock:
            self.weight = target

    def play(self, plan):
        """Play every planned frame and retain the final pose between requests.

        The caller supplies hardware-bounded IK solutions. The first target is
        one model period after the measured starting pose; no stale-prefix skip
        or maximum displacement, latency, tilt, or tracking-error gate is used.
        """
        self.check()
        self._standing()
        if plan.get("side", self.side) != self.side:
            raise ValueError("Trajectory belongs to a different arm")
        joints = np.asarray(plan["joints"], dtype=float)
        hands = np.asarray(plan["hands"], dtype=float)
        fps = float(plan.get("fps", self.fps))
        if (joints.ndim != 2 or joints.shape[1] != 5 or len(joints) == 0
                or hands.shape != (len(joints), 6) or not np.isfinite(joints).all()
                or not np.isfinite(hands).all() or not np.isfinite(fps) or fps <= 0):
            raise ValueError("Invalid arm/hand trajectory schema")
        _, motors, _ = self.sensors.feedback()
        points = np.vstack((motors[list(ARM_IDS[self.side])], joints))
        started = time.monotonic()
        with self.lock:
            self.trajectory = points, started, fps
        for index, hand in enumerate(hands):
            self.check()
            deadline = started + (index + 1) / fps
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.check()
                time.sleep(min(1 / STREAM_HZ, remaining))
            with self.lock:
                if self.closed:
                    raise RuntimeError("Arm session is closed")
                self.hands.write(self.side, hand)
                self.hand_target = hand.copy()
                self.hand_sent_at = time.monotonic()
        self.check()
        with self.lock:
            self.command[self.side] = joints[-1].copy()
            self.trajectory = None

    def close(self):
        """Release arm authority and restore the previous service; never open hands."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.hand_target = None
            self.trajectory = None
        errors = []
        try:
            if self.thread is not None and self.thread.is_alive() and self.fault is None:
                self._ramp(0.0, RAMP_OUT_S)
        except BaseException as exc:
            errors.append(exc)
        finally:
            self.stop_event.set()
            if self.thread is not None:
                self.thread.join()
            if self.publisher is not None:
                try:
                    self.weight = 0.0
                    self._publish()
                except Exception as exc:
                    errors.append(exc)
                finally:
                    try:
                        self.publisher.Close()
                    except Exception as exc:
                        errors.append(exc)
            if self.hands is not None:
                try:
                    self.hands.close()
                except Exception as exc:
                    errors.append(exc)
            if self.restore_service:
                self.restore_service = False
                try:
                    code = self.services.ServiceSwitch(ARM_SERVICE, True)
                    if code != 0 or self._service_status() != 0:
                        raise RuntimeError(f"Cannot restore {ARM_SERVICE}: {code}")
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]
