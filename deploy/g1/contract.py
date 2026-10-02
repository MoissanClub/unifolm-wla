"""WBT observation/action contract. Arrays use SI units and RGB images.

State slot 'base_rotvec' is gravity + angular velocity in this revision.
The absolute base pose is a separate anchor; it is never put in that slot.
"""
from __future__ import annotations

import numpy as np

from unifolm_wla.dataloader.multi_source_dataset.action_mapping import (
    SLICES, STATE_DIM, UNIFIED_DIM, map_state,
)
from unifolm_wla.dataloader.multi_source_dataset.config import DatasetSourceConfig
from unifolm_wla.dataloader.multi_source_dataset.se3_utils import (
    pose_to_se3_from_format,
)

PROTOCOL = "unifolm-g1-wbt-v1"
IMAGE_SIZE = (336, 448)  # H,W; upstream unitree.yaml training resolution
ROLES = ("head_left", "cam_wrist_left", "cam_wrist_right")
STATE_WIDTHS = {
    "left_ee_pose": 6, "right_ee_pose": 6,
    "left_fig6d": 6, "right_fig6d": 6, "waist_joint": 3,
    "left_leg": 6, "right_leg": 6, "base_rot": 6,
}
ACTION_FIELDS = (
    "left_xyz_rotvec", "left_fig6d", "right_xyz_rotvec", "right_fig6d",
    "waist_joint", "base_vx_vy", "base_vw", "base_rotvec", "height",
    "left_leg_joint", "right_leg_joint",
)
SOURCE = DatasetSourceConfig(
    name="UnifoLM_WBT", data_path="", robot_type="unitree", source_fps=30,
    arm_type="dual_with_legs", ee_format="xyz_rpy",
)


def array(value, shape, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite {shape}, got {value.shape}")
    return value


def action_mask(profile="wbt"):
    if profile not in ("wbt", "stationary_arms"):
        raise ValueError(f"Unknown action profile: {profile}")
    mask = np.zeros(UNIFIED_DIM, dtype=bool)
    fields = ACTION_FIELDS if profile == "wbt" else ACTION_FIELDS[:4]
    for key in fields:
        mask[SLICES[key]] = True
    return mask


def validate_observation(obs, allow_missing_cameras=False):
    if obs.get("protocol") != PROTOCOL:
        raise ValueError("Observation protocol mismatch")
    if not isinstance(obs.get("instruction"), str) or not obs["instruction"].strip():
        raise ValueError("A nonempty task instruction is required")
    if not isinstance(obs.get("frame_id"), str):
        raise ValueError("frame_id must be a string")
    state = {key: array(obs["state"][key], (width,), key) for key, width in STATE_WIDTHS.items()}
    for side in ("left", "right"):
        hand = state[f"{side}_fig6d"]
        if np.any((hand < -0.001) | (hand > 1.001)):
            raise ValueError("BrainCo finger positions must be in [0, 1]")
    if abs(np.linalg.norm(state["base_rot"][:3]) - 1) > 0.1:
        raise ValueError("base_rot[:3] must be projected unit gravity, not position")
    if "base_pose" in obs:
        pose = array(obs["base_pose"], (7,), "base_pose xyz+xyzw")
        if abs(np.linalg.norm(pose[3:]) - 1) > 0.01:
            raise ValueError("Invalid base-pose quaternion")
    images = obs["images"]
    if set(images) - set(ROLES) or "head_left" not in images:
        raise ValueError(f"Expected image roles from {ROLES}, including head_left")
    if not allow_missing_cameras and set(images) != set(ROLES):
        raise ValueError("Missing wrist cameras; select --allow-missing-cameras for an explicit ablation")
    for role, img in images.items():
        img = np.asarray(img)
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"{role}: expected HxWx3 uint8 RGB")
        if not (32 <= img.shape[0] <= 1080 and 32 <= img.shape[1] <= 1920):
            raise ValueError(f"{role}: unsupported image resolution")
    return state


class WBTAdapter:
    def __init__(self, statistics, key="UnifoLM_WBT", allow_missing_cameras=False, profile="wbt"):
        if key not in statistics:
            raise ValueError(f"Unknown statistics key {key!r}; available: {list(statistics)}")
        self.key = key
        self.allow_missing_cameras = allow_missing_cameras
        self.profile = profile
        self.action_mask = action_mask(profile)
        stats = statistics[key]
        self.norm = {}
        for kind, width in (("state", STATE_DIM), ("action", UNIFIED_DIM)):
            for field in ("offset", "scale"):
                self.norm[f"{kind}_{field}"] = array(stats[kind][field], (width,), f"{kind}.{field}")
            if np.any(self.norm[f"{kind}_scale"] <= 0):
                raise ValueError(f"Nonpositive {kind} normalization scale")

    def example(self, obs):
        state = validate_observation(obs, self.allow_missing_cameras)
        if obs.get("unnorm_key", self.key) != self.key:
            raise ValueError("Client/server normalization keys differ")
        raw, mask = map_state(SOURCE, state)
        roles = [r for r in ROLES if r in obs["images"]]
        return {
            "image": [np.asarray(obs["images"][r]) for r in roles],
            "image_roles": roles, "lang": obs["instruction"],
            "state": (raw - self.norm["state_offset"]) / (self.norm["state_scale"] + 1e-8),
            "state_mask": mask, "action_mask": self.action_mask,
            "arm_type": "dual_with_legs", "robot_type": "unitree",
        }

    def decode(self, normalized, obs):
        pred = np.asarray(normalized, dtype=np.float32)
        if pred.ndim != 2 or pred.shape[1] != UNIFIED_DIM or not (1 <= len(pred) <= 120):
            raise ValueError("Expected an action chunk [T,54], 1 <= T <= 120")
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite model output")
        raw = pred * self.norm["action_scale"] + self.norm["action_offset"]
        result = {"normalized": pred, "raw": raw, "action_profile": self.profile}
        for side in ("left", "right"):
            anchor = pose_to_se3_from_format(np.asarray(obs["state"][f"{side}_ee_pose"]), "xyz_rpy")
            relative = pose_to_se3_from_format(raw[:, SLICES[f"{side}_xyz_rotvec"]], "xyz_rvec")
            # Every prediction is relative to ONE observation anchor, not the previous prediction.
            result[f"{side}_T"] = (anchor @ relative).astype(np.float32)
            result[f"{side}_hand"] = raw[:, SLICES[f"{side}_fig6d"]]
        result["base_relative"] = raw[:, SLICES["base_rotvec"]]
        if "base_pose" in obs:
            anchor = pose_to_se3_from_format(np.asarray(obs["base_pose"]), "xyz_quat")
            result["base_T"] = anchor @ pose_to_se3_from_format(result["base_relative"], "xyz_rvec")
        result["base_command"] = np.concatenate([
            raw[:, SLICES["base_vx_vy"]], raw[:, SLICES["base_vw"]], raw[:, SLICES["height"]],
        ], axis=1)
        result["waist"] = raw[:, SLICES["waist_joint"]]
        result["legs"] = np.concatenate([raw[:, SLICES["left_leg_joint"]], raw[:, SLICES["right_leg_joint"]]], axis=1)
        return result
