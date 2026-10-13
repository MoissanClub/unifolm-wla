"""Read-only PC2 sensors and calibrated 23-DoF FK/IK. No publisher in this module."""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .contract import PROTOCOL, ROLES, array
from .semantics import native_hand_to_policy, pelvis_pose_to_world

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

    def fit(self, motors, side, target):
        """Fit a policy pose to this arm; report error without a task cutoff.

        Bounds come from the physical URDF, without an added joint margin.
        A 5-DoF arm cannot generally satisfy a complete 6D policy pose.
        """
        target = transform(target)
        ids = list(ARM_IDS[side])
        indices = [self.indices[i] for i in ids]
        lo = self.model.lowerPositionLimit[indices]
        hi = self.model.upperPositionLimit[indices]
        original = np.asarray(motors, dtype=float)

        def residual(arm):
            q = original.copy()
            q[ids] = arm
            pose = self.fk(q, side)
            return np.r_[pose[:3, 3]-target[:3, 3],
                         Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec()]

        solution = least_squares(residual, np.clip(original[ids], lo, hi), bounds=(lo, hi))
        error = residual(solution.x)
        return solution.x, {"position_error_m": float(np.linalg.norm(error[:3])),
                            "rotation_error_rad": float(np.linalg.norm(error[3:])),
                            "converged": bool(solution.success)}

    def ik(self, motors, side, target):
        return self.fit(motors, side, target)[0]


class Sensors:
    def __init__(self, interface, kinematics, wrist_left=None, wrist_right=None,
                 external_images=False, initialize_dds=True):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_, SportModeState_
        self.kin = kinematics
        self.lock = threading.Lock()
        self.low_lock = threading.Lock()
        self.low = None
        self.hands = {}
        self.stamps = {}
        self.subscribers = []
        self.cameras = {}
        self.pipeline = None
        self.pipeline_started = False
        self.odom = None
        self.reference = None
        try:
            if initialize_dds:
                ChannelFactoryInitialize(0, interface)

            # Poll on demand. A Python callback deserializes all 500 Hz messages
            # even when the application only needs a fresh sample at capture time.
            self.low_sub = ChannelSubscriber("rt/lowstate", LowState_)
            self.subscribers.append(self.low_sub)
            self.low_sub.Init()
            def odom_callback(msg):
                with self.lock:
                    self.odom = msg
                    self.stamps["odometry"] = time.monotonic()

            sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
            self.subscribers.append(sub)
            sub.Init(odom_callback)
            for side in ARM_IDS:
                def callback(msg, side=side):
                    if len(msg.states) != 6:
                        return
                    with self.lock:
                        self.hands[side] = msg
                        self.stamps[side] = time.monotonic()
                sub = ChannelSubscriber(f"rt/brainco/{side}/state", MotorStates_)
                self.subscribers.append(sub)
                sub.Init(callback)
            while True:
                if self.low is None:
                    self.low = self.low_sub.Read()
                with self.lock:
                    ready = self.low is not None and len(self.hands) == 2 and self.odom is not None
                if ready:
                    break
                time.sleep(0.02)
            if external_images:
                return
            import cv2
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            self.pipeline.start(config)
            self.pipeline_started = True
            for role, path in zip(ROLES[1:], (wrist_left, wrist_right)):
                if path:
                    camera = cv2.VideoCapture(path)
                    self.cameras[role] = camera
                    if not camera.isOpened():
                        raise RuntimeError(f"Cannot open {role} at {path}")
        except BaseException as initialization_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise initialization_error from cleanup_error
            raise

    def feedback(self):
        with self.low_lock:
            low = self.low_sub.Read()
        if low is None:
            raise RuntimeError("No robot feedback received")
        with self.lock:
            self.low = low
            self.stamps["low"] = time.monotonic()
            hands = {s: np.array([v.q for v in msg.states], dtype=np.float32)
                     for s, msg in self.hands.items()}
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
            # Inference can take seconds. Discard frames queued during that gap
            # before waiting for a new frame, keeping capture_start conservative.
            self.pipeline.poll_for_frames()
            frames = self.pipeline.wait_for_frames()
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
        base_pose = self.base_pose()
        quat = np.asarray(low.imu_state.quaternion, dtype=float)[[1, 2, 3, 0]]  # SDK wxyz -> xyzw
        gravity = Rotation.from_quat(quat).inv().apply([0.0, 0.0, -1.0])
        state = {"waist_joint": np.array([motors[12], 0, 0], dtype=np.float32),
                 "left_leg": motors[:6], "right_leg": motors[6:12],
                 "base_rot": np.r_[gravity, low.imu_state.gyroscope]}
        for side in ARM_IDS:
            pose = pelvis_pose_to_world(base_pose, self.kin.fk(motors, side))
            state[f"{side}_ee_pose"] = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")]
            state[f"{side}_fig6d"] = native_hand_to_policy(hands[side])
        return {"protocol": PROTOCOL, "frame_id": uuid.uuid4().hex,
                "instruction": instruction, "unnorm_key": "UnifoLM_WBT", "state": state,
                "images": images, "captured_monotonic": capture_start,
                "base_pose": base_pose,
                "base_frame": "standing_task_reference",
                "hand_convention": "wbt_fig6d_one_minus_native",
                "capture_ms": (time.monotonic()-capture_start)*1000,
                "feedback_age_s": self.feedback_ages(),
                "motors": motors, "mode_machine": int(low.mode_machine)}

    def feedback_ages(self):
        with self.lock:
            return {s: time.monotonic()-stamp for s, stamp in self.stamps.items()}

    def base_pose(self):
        """Standing task frame: initial x/y/yaw zero, z from body_height.

        Raw odometry z on PC2 has an arbitrary startup origin. The firmware's
        body_height supplies the floor reference used by the dataset. Preserve
        a fixed transform for this session rather than resetting each capture.
        """
        with self.lock:
            odom = self.odom
        rotation = Rotation.from_quat(np.asarray(odom.imu_state.quaternion)[[1, 2, 3, 0]])
        T = np.eye(4)
        T[:3, :3] = rotation.as_matrix()
        T[:3, 3] = odom.position
        if self.reference is None:
            self.reference = np.eye(4)
            self.reference[:3, :3] = Rotation.from_euler("z", -rotation.as_euler("xyz")[2]).as_matrix()
            self.reference[:3, 3] = -self.reference[:3, :3] @ T[:3, 3]
            self.reference[2, 3] += float(odom.body_height)
        pose = self.reference @ T
        return np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]

    def close(self):
        # Detach ownership before closing so repeated cleanup is harmless. A
        # failing device must not prevent the other cameras/subscribers closing.
        pipeline, self.pipeline = self.pipeline, None
        pipeline_started, self.pipeline_started = self.pipeline_started, False
        cameras, self.cameras = self.cameras, {}
        subscribers, self.subscribers = self.subscribers, []
        errors = []
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError as exc:
                # RealSense raises when initialization did not reach start().
                if pipeline_started:
                    errors.append(exc)
            except BaseException as exc:
                errors.append(exc)
        for camera in cameras.values():
            try:
                camera.release()
            except BaseException as exc:
                errors.append(exc)
        for sub in subscribers:
            try:
                sub.Close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
