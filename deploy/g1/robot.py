"""Read-only PC2 sensors and calibrated 23-DoF FK/IK. No publisher in this module."""
from __future__ import annotations

import copy
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .contract import PROTOCOL, ROLES, array

ARM_IDS = {"left": (15, 16, 17, 18, 19), "right": (22, 23, 24, 25, 26)}
LEG_NAMES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM_NAMES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
JOINTS = {i + start: f"{side}_{name}_joint" for side, start in (("left", 0), ("right", 6))
          for i, name in enumerate(LEG_NAMES)}
JOINTS[12] = "waist_yaw_joint"
JOINTS.update({idx: f"{side}_{name}_joint" for side in ARM_IDS for idx, name in zip(ARM_IDS[side], ARM_NAMES)})


def transform(value):
    T = array(value, (4, 4), "wrist_T_policy")
    if not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Invalid rigid transform")
    if not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5):
        raise ValueError("Transform contains a reflection")
    return T


class Kinematics23:
    def __init__(self, urdf, calibration):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(Path(urdf).resolve()))
        self.data = self.model.createData()
        if self.model.nq != 23:
            raise ValueError(f"Expected fixed-base 23-DoF URDF, got nq={self.model.nq}")
        self.indices = {}
        for motor, name in JOINTS.items():
            if not self.model.existJointName(name):
                raise ValueError(f"URDF missing {name}")
            self.indices[motor] = self.model.joints[self.model.getJointId(name)].idx_q
        self.wrist_ids = {s: self.model.getJointId(f"{s}_wrist_roll_joint") for s in ARM_IDS}
        self.offset = {s: transform(calibration["wrist_T_policy"][s]) for s in ARM_IDS}

    def full_q(self, motors):
        q = self.pin.neutral(self.model)
        for motor, index in self.indices.items():
            q[index] = motors[motor]
        return q

    def fk(self, motors, side):
        self.pin.forwardKinematics(self.model, self.data, self.full_q(motors))
        return self.data.oMi[self.wrist_ids[side]].homogeneous.copy() @ self.offset[side]

    def ik(self, motors, side, target, pos_tol=0.015, rot_tol=0.15):
        target = transform(target)
        ids = list(ARM_IDS[side])
        indices = [self.indices[i] for i in ids]
        lo = self.model.lowerPositionLimit[indices] + 0.02
        hi = self.model.upperPositionLimit[indices] - 0.02
        original = np.asarray(motors, dtype=float)

        def residual(arm):
            q = original.copy()
            q[ids] = arm
            pose = self.fk(q, side)
            return np.r_[(pose[:3, 3]-target[:3, 3])/pos_tol,
                         Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec()/rot_tol]

        solution = least_squares(residual, np.clip(original[ids], lo, hi), bounds=(lo, hi), max_nfev=50)
        error = residual(solution.x)
        position_error, rotation_error = np.linalg.norm(error[:3])*pos_tol, np.linalg.norm(error[3:])*rot_tol
        if not solution.success or position_error > pos_tol or rotation_error > rot_tol:
            raise ValueError(f"{side} unreachable on 5-DoF arm: {position_error:.3f} m, {rotation_error:.3f} rad")
        return solution.x


class Sensors:
    def __init__(self, interface, kinematics, wrist_left=None, wrist_right=None,
                 external_images=False, initialize_dds=True):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
        self.kin = kinematics
        self.lock = threading.Lock()
        self.low = None
        self.hands = {}
        self.stamps = {}
        self.subscribers = []
        self.cameras = {}
        self.pipeline = None
        if initialize_dds:
            ChannelFactoryInitialize(0, interface)

        def low_callback(msg):
            with self.lock:
                self.low = copy.deepcopy(msg)
                self.stamps["low"] = time.monotonic()

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(low_callback, 1)
        self.subscribers.append(sub)
        for side in ARM_IDS:
            def callback(msg, side=side):
                if len(msg.states) != 6:
                    return
                with self.lock:
                    self.hands[side] = np.array([v.q for v in msg.states], dtype=np.float32)
                    self.stamps[side] = time.monotonic()
            sub = ChannelSubscriber(f"rt/brainco/{side}/state", MotorStates_)
            sub.Init(callback, 1)
            self.subscribers.append(sub)
        start = time.monotonic()
        while True:
            with self.lock:
                ready = self.low is not None and len(self.hands) == 2
            if ready:
                break
            if time.monotonic() - start > 5:
                self.close()
                raise RuntimeError("Missing lowstate or BrainCo hand feedback")
            time.sleep(0.02)
        if external_images:
            return
        try:
            import cv2
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            self.pipeline.start(config)
            for role, path in zip(ROLES[1:], (wrist_left, wrist_right)):
                if path:
                    camera = cv2.VideoCapture(path)
                    self.cameras[role] = camera
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not camera.isOpened():
                        raise RuntimeError(f"Cannot open {role} at {path}")
        except Exception:
            self.close()
            raise

    def feedback(self, max_age=0.2):
        with self.lock:
            now = time.monotonic()
            if any(now-t > max_age for t in self.stamps.values()) or len(self.stamps) != 3:
                raise RuntimeError("Stale robot/hand feedback")
            low, hands = self.low, {s: q.copy() for s, q in self.hands.items()}
        if int(low.mode_machine) not in (1, 4) or int(low.mode_pr) != 0:
            raise RuntimeError("Expected G1 23-DoF with PR joint feedback")
        motors = np.array([m.q for m in low.motor_state], dtype=np.float64)
        if not np.isfinite(motors).all():
            raise RuntimeError("Invalid motor feedback")
        return low, motors, hands

    def observe(self, instruction, images=None):
        import cv2
        capture_start = time.monotonic()
        if images is None:
            if self.pipeline is None:
                raise ValueError("Supply fresh RGB images from the existing camera owner")
            frames = self.pipeline.wait_for_frames(1000)
            image = frames.get_color_frame()
            if not image:
                raise RuntimeError("No RealSense color frame")
            images = {"head_left": np.asanyarray(image.get_data()).copy()}
            for role, camera in self.cameras.items():
                ok, bgr = camera.read()
                if not ok:
                    raise RuntimeError(f"Missing image: {role}")
                images[role] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        low, motors, hands = self.feedback()
        if time.monotonic()-capture_start > 0.2:
            raise RuntimeError("Camera acquisition exceeds 200 ms; fix buffering/synchronization")
        quat = np.asarray(low.imu_state.quaternion, dtype=float)[[1, 2, 3, 0]]  # SDK wxyz -> xyzw
        gravity = Rotation.from_quat(quat).inv().apply([0.0, 0.0, -1.0])
        state = {"waist_joint": np.array([motors[12], 0, 0], dtype=np.float32),
                 "left_leg": motors[:6], "right_leg": motors[6:12],
                 "base_rot": np.r_[gravity, low.imu_state.gyroscope]}
        for side in ARM_IDS:
            pose = self.kin.fk(motors, side)
            state[f"{side}_ee_pose"] = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")]
            state[f"{side}_fig6d"] = hands[side]
        return {"protocol": PROTOCOL, "frame_id": uuid.uuid4().hex,
                "instruction": instruction, "unnorm_key": "UnifoLM_WBT", "state": state,
                "images": images, "captured_monotonic": capture_start,
                "motors": motors, "mode_machine": int(low.mode_machine)}

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
        for camera in self.cameras.values():
            camera.release()
        for sub in self.subscribers:
            sub.Close()
