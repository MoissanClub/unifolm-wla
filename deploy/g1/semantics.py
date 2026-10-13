"""Conversions verified against the released WBT BrainCo dataset.

DDS finger positions use 0=open, 1=closed. WBT ``fig6d`` uses the inverse
convention. WBT ``ee_pose_gripper_base`` is expressed in the reference frame
containing ``state_base_pose``; it is not a pose relative to the pelvis.
See reports/2026-10-13-dataset-semantics.json for the numerical evidence.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def native_hand_to_policy(native):
    """Convert [...,6] BrainCo DDS positions to WBT fig6d, preserving order."""
    native = np.asarray(native, dtype=np.float32)
    if native.ndim < 1 or native.shape[-1] != 6 or not np.isfinite(native).all():
        raise ValueError("Expected finite BrainCo finger positions [...,6]")
    return 1.0 - native


def policy_hand_to_native(policy):
    """Convert [...,6] WBT fig6d targets to BrainCo DDS positions."""
    return native_hand_to_policy(policy)


def base_pose_to_transform(base_pose):
    """Return world_T_pelvis from a base pose [x,y,z,qx,qy,qz,qw]."""
    pose = np.asarray(base_pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite base pose [x,y,z,qx,qy,qz,qw]")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def pelvis_pose_to_world(base_pose, pelvis_T_ee):
    """Map pelvis-frame FK pose(s) [...,4,4] to the WBT reference frame."""
    return base_pose_to_transform(base_pose) @ np.asarray(pelvis_T_ee)


def world_pose_to_pelvis(base_pose, world_T_ee):
    """Map decoded WBT pose(s) [...,4,4] into the IK pelvis frame."""
    return np.linalg.inv(base_pose_to_transform(base_pose)) @ np.asarray(world_T_ee)
