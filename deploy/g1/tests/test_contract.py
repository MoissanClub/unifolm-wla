import subprocess
import sys

import numpy as np
import pytest

from deploy.g1.contract import WBTAdapter, action_mask, STATE_DIM, SLICES
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import STATE_SLICES


def test_mapping_does_not_import_training_stack():
    subprocess.run([sys.executable, "-c", "import sys; from deploy.g1.contract import WBTAdapter; "
                    "assert 'torch' not in sys.modules; assert 'lerobot' not in sys.modules"], check=True)


def test_wbt_state_is_gravity_not_base_position(observation, statistics):
    example = WBTAdapter(statistics).example(observation)
    assert example["state"].shape == (STATE_DIM,)
    np.testing.assert_allclose(example["state"][STATE_SLICES["base_rotvec"]], [0, 0, -1, .1, .2, .3])
    assert example["state_mask"].sum() == 51
    assert example["action_mask"].sum() == 49
    assert not example["action_mask"][SLICES["left_gripper"]].any()
    assert example["action_mask"][SLICES["left_fig6d"]].all()


def test_same_anchor_for_every_relative_action(observation, statistics):
    observation["state"]["left_ee_pose"][5] = np.pi/2
    pred = np.zeros((2, 54), np.float32)
    pred[:, 0] = [0.1, 0.2]
    pred[:, 35] = [0.5, 1.0]
    result = WBTAdapter(statistics).decode(pred, observation)
    np.testing.assert_allclose(result["left_T"][:, :3, 3], [[.3, .35, .2], [.3, .45, .2]], atol=1e-6)
    np.testing.assert_allclose(result["base_T"][:, :3, 3], [[10.5, 20, .8], [11, 20, .8]], atol=1e-6)


def test_missing_left_wrist_preserves_right_label(observation, statistics):
    del observation["images"]["cam_wrist_left"]
    with pytest.raises(ValueError, match="Missing wrist"):
        WBTAdapter(statistics).example(observation)
    example = WBTAdapter(statistics, allow_missing_cameras=True).example(observation)
    assert example["image_roles"] == ["head_left", "cam_wrist_right"]


@pytest.mark.parametrize("key,value", [("left_fig6d", [1.2]*6), ("base_rot", [0]*6), ("left_leg", [0]*5)])
def test_invalid_observation_rejected(observation, statistics, key, value):
    observation["state"][key] = value
    with pytest.raises(ValueError):
        WBTAdapter(statistics).example(observation)


def test_invalid_prediction_and_stats(observation, statistics):
    with pytest.raises(ValueError, match="Unknown statistics"):
        WBTAdapter(statistics, "UnifoLM_G1_Dex1")
    with pytest.raises(ValueError, match="Nonfinite"):
        WBTAdapter(statistics).decode(np.full((30, 54), np.nan), observation)
    statistics["UnifoLM_WBT"]["action"]["scale"][0] = 0
    with pytest.raises(ValueError, match="Nonpositive"):
        WBTAdapter(statistics)


def test_stationary_arm_mask():
    assert action_mask("stationary_arms").sum() == 24
    with pytest.raises(ValueError):
        action_mask("something_else")


def test_msgpack_roundtrip(observation):
    from model_server.tools.msgpack_numpy import packb, unpackb
    result = unpackb(packb(observation))
    np.testing.assert_array_equal(result["images"]["head_left"], observation["images"]["head_left"])
    np.testing.assert_array_equal(result["state"]["base_rot"], observation["state"]["base_rot"])
