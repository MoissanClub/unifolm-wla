import numpy as np
from scipy.spatial.transform import Rotation

from deploy.g1.semantics import (
    native_hand_to_policy, policy_hand_to_native,
    pelvis_pose_to_world, world_pose_to_pelvis,
)


def test_released_brainco_finger_sample():
    # Official dataset file-001, episode165/frame0: native hand_state and fig6d.
    native = np.array([0, 0.779, 0, 0, 0, 0], dtype=np.float32)
    policy = np.array([1, 0.221, 1, 1, 1, 1], dtype=np.float32)
    np.testing.assert_allclose(native_hand_to_policy(native), policy, atol=3e-8)
    np.testing.assert_allclose(policy_hand_to_native(policy), native, atol=3e-8)


def test_released_brainco_action_chunk_preserves_finger_order():
    policy = np.array([[1, .2, 1, 1, 1, 1], [.1, .2, .3, .4, .5, .6]])
    expected = np.array([[0, .8, 0, 0, 0, 0], [.9, .8, .7, .6, .5, .4]])
    np.testing.assert_allclose(policy_hand_to_native(policy), expected)


def test_world_reference_includes_body_height_and_heading():
    # Heading90deg maps pelvis-forward to world-left; height must not be lost.
    base_pose = [2, 3, .8, 0, 0, np.sqrt(.5), np.sqrt(.5)]
    local = np.eye(4)
    local[:3, 3] = [.4, .2, .1]
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler('z', np.pi/2).as_matrix()
    expected[:3, 3] = [1.8, 3.4, .9]
    np.testing.assert_allclose(pelvis_pose_to_world(base_pose, local), expected, atol=1e-14)
    np.testing.assert_allclose(world_pose_to_pelvis(base_pose, expected), local, atol=1e-14)


def test_decoded_world_chunk_uses_one_base_reference():
    base_pose = [0, 0, .75, 0, 0, 0, 1]
    chunk = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    chunk[:, :3, 3] = [[.2, .2, .8], [.3, .2, .9], [.4, .2, 1.0]]
    local = world_pose_to_pelvis(base_pose, chunk)
    np.testing.assert_allclose(local[:, :3, 3], [[.2, .2, .05], [.3, .2, .15], [.4, .2, .25]])
