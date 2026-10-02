"""Conservative stationary-arm retargeting gate; never drives model leg outputs."""
from __future__ import annotations

import time
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import PROTOCOL, array
from .robot import ARM_IDS, transform


def choose_target(reply, observation, motors, kinematics, limits, now=None):
    now = time.monotonic() if now is None else now
    age = now - observation["captured_monotonic"]
    if not 0 <= age <= limits["max_observation_age_s"]:
        raise ValueError(f"Stale action: observation age {age:.3f}s")
    if reply.get("protocol") != PROTOCOL or reply.get("frame_id") != observation["frame_id"]:
        raise ValueError("Action belongs to a different observation/protocol")
    # Expired prefix is never replayed. Reserve time for reaching the selected target.
    index = int(np.ceil((age + limits["move_duration_s"]) * 30))
    raw = np.asarray(reply["raw"])
    if raw.ndim != 2 or raw.shape[1] != 54 or not np.isfinite(raw).all():
        raise ValueError("Invalid action chunk")
    horizon = len(raw)
    if index >= horizon:
        raise ValueError("Inference consumed the action horizon; use a faster inference host")
    base = array(reply["base_command"], (horizon, 4), "base_command")
    relative = array(reply["base_relative"], (horizon, 6), "base_relative")
    waist = array(reply["waist"], (horizon, 3), "waist")
    legs = array(reply["legs"], (horizon, 12), "legs")
    profile = reply.get("action_profile")
    if profile not in ("wbt", "stationary_arms"):
        raise ValueError("Unknown action profile")
    if profile == "wbt":
        # Reject plans dependent on stepping, squatting, or unavailable waist joints.
        if np.max(np.abs(base[:index+1, :3])) > limits["max_base_speed"]:
            raise ValueError("Policy requested locomotion; this executor is stationary")
        if np.max(np.linalg.norm(relative[:index+1, :3], axis=1)) > limits["max_base_translation_m"]:
            raise ValueError("Policy requested base translation")
        if np.max(np.linalg.norm(relative[:index+1, 3:], axis=1)) > limits["max_base_rotation_rad"]:
            raise ValueError("Policy requested base rotation")
        if np.max(np.abs(waist[:index+1] - [motors[12], 0, 0])) > limits["max_lower_joint_delta_rad"]:
            raise ValueError("Policy requested unsupported waist movement")
        if np.max(np.abs(legs[:index+1] - motors[:12])) > limits["max_lower_joint_delta_rad"]:
            raise ValueError("Policy relies on lower-body motion")
    if np.max(np.abs(motors[:13] - np.asarray(observation["motors"])[:13])) > 0.05:
        raise ValueError("Base/waist moved since the observation")
    targets = {}
    for side in ARM_IDS:
        poses = array(reply[f"{side}_T"], (horizon, 4, 4), side)
        pose = transform(poses[index])
        current = kinematics.fk(motors, side)
        distance = np.linalg.norm(pose[:3, 3]-current[:3, 3])
        angle = Rotation.from_matrix(current[:3, :3].T @ pose[:3, :3]).magnitude()
        if distance > limits["max_cartesian_step_m"] or angle > limits["max_rotation_step_rad"]:
            raise ValueError(f"{side} prediction exceeds Cartesian step limit")
        q = kinematics.ik(motors, side, pose)
        if np.max(np.abs(q-motors[list(ARM_IDS[side])])) > limits["max_joint_step_rad"]:
            raise ValueError(f"{side} IK exceeds joint step limit")
        # Hand outputs are logged but held during free-space arm evaluation.
        array(reply[f"{side}_hand"], (horizon, 6), f"{side}_hand")
        targets[side] = q
    if np.linalg.norm(reply["left_T"][index, :3, 3]-reply["right_T"][index, :3, 3]) < limits["min_hand_separation_m"]:
        raise ValueError("Hands too close")
    return targets, index
