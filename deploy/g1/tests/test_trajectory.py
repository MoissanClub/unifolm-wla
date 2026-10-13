import numpy as np
import pytest

from deploy.g1.contract import PROTOCOL
from deploy.g1.robot import ARM_IDS
from deploy.g1.trajectory import plan_chunk


class FakeKinematics:
    def __init__(self, errors=None):
        self.seeds = []
        self.targets = []
        self.errors = errors or {"position_error_m": 0., "rotation_error_rad": 0., "converged": True}

    def fit(self, motors, side, pose):
        self.seeds.append(motors.copy())
        self.targets.append(pose.copy())
        return motors[list(ARM_IDS[side])] + 0.01, self.errors


def reply_for(observation, horizon=30):
    poses = np.broadcast_to(np.eye(4), (horizon, 4, 4)).copy()
    poses[:, :3, 3] = [2., 3.4, .9]
    return {
        "protocol": PROTOCOL, "frame_id": observation["frame_id"], "raw": np.zeros((horizon, 54)),
        "left_T": poses.copy(), "right_T": poses.copy(),
        "left_hand": np.tile([1., .2, 1., 1., 1., 1.], (horizon, 1)),
        "right_hand": np.zeros((horizon, 6)), "base_command": np.zeros((horizon, 4)),
        "base_relative": np.zeros((horizon, 6)), "waist": np.zeros((horizon, 3)),
        "legs": np.zeros((horizon, 12)),
    }


def test_world_reference_and_native_hand_commands(observation):
    observation["base_pose"] = np.array([2, 3, .8, 0, 0, np.sqrt(.5), np.sqrt(.5)])
    kin = FakeKinematics()
    plan = plan_chunk(reply_for(observation), observation, kin)
    np.testing.assert_allclose(kin.targets[0][:3, 3], [.4, 0, .1], atol=1e-7)
    np.testing.assert_allclose(plan["hands"][0], [0., .8, 0., 0., 0., 0.])
    assert plan["joints"].shape == (30, 5)
    assert plan["motors"].shape == (30, 35)


def test_old_observation_keeps_entire_chunk_and_warmstarts(observation):
    observation["captured_monotonic"] = -1000000
    kin = FakeKinematics()
    plan = plan_chunk(reply_for(observation), observation, kin, fps=30)
    assert len(kin.targets) == 30
    assert plan["time_s"][0] == 1 / 30
    assert plan["time_s"][-1] == 1
    np.testing.assert_allclose(kin.seeds[1][list(ARM_IDS["left"])], plan["joints"][0])
    np.testing.assert_allclose(plan["motors"][:, list(ARM_IDS["right"])], 0)
    np.testing.assert_allclose(observation["motors"], 0)


def test_wrong_frame_fails_before_ik(observation):
    reply = reply_for(observation)
    reply["frame_id"] = "unrelated-frame"
    kin = FakeKinematics()
    with pytest.raises(ValueError, match="different observation"):
        plan_chunk(reply, observation, kin)
    assert not kin.targets


def test_solver_error_and_unexecuted_body_motion_are_diagnostics(observation):
    reply = reply_for(observation)
    reply["base_command"][:, 0] = .4
    reply["base_relative"][:, 0] = .5
    reply["legs"][:, 0] = .6
    kin = FakeKinematics({"position_error_m": .1, "rotation_error_rad": .3, "converged": False})
    plan = plan_chunk(reply, observation, kin)
    np.testing.assert_allclose(plan["errors"]["position_error_m"], .1)
    np.testing.assert_allclose(plan["errors"]["rotation_error_rad"], .3)
    assert not plan["errors"]["converged"].any()
    assert plan["diagnostics"]["base_translation_max_m"] == .5
    assert plan["diagnostics"]["leg_change_max_rad"] == pytest.approx(.6)


@pytest.mark.parametrize("field", ["raw", "left_T", "right_hand", "legs"])
def test_nonfinite_chunk_is_invalid_schema(observation, field):
    reply = reply_for(observation)
    reply[field].flat[0] = np.nan
    with pytest.raises(ValueError):
        plan_chunk(reply, observation, FakeKinematics())


def test_offline_cli_saves_provenance_without_sensor_initialization(tmp_path, observation, calibration, monkeypatch):
    import json
    import sys
    from deploy.g1 import robot, trajectory
    from model_server.tools.msgpack_numpy import packb, unpackb

    observation_path = tmp_path / "observation.msgpack"
    prediction_path = tmp_path / "prediction.msgpack"
    calibration_path = tmp_path / "calibration.json"
    output = tmp_path / "plan.msgpack"
    observation_path.write_bytes(packb(observation))
    prediction_path.write_bytes(packb(reply_for(observation)))
    calibration_path.write_text(json.dumps(calibration))
    monkeypatch.setattr(robot, "Kinematics23", lambda *args: FakeKinematics())

    def unexpected(*args, **kwargs):
        raise AssertionError("Offline planner must not open sensors")

    monkeypatch.setattr(robot, "Sensors", unexpected)
    monkeypatch.setattr(sys, "argv", ["trajectory", "--observation", str(observation_path),
                                     "--prediction", str(prediction_path),
                                     "--calibration", str(calibration_path), "--urdf", "unused.urdf",
                                     "--out", str(output)])
    trajectory.main()
    result = unpackb(output.read_bytes())
    assert result["joints"].shape == (30, 5)
    assert result["provenance"]["execution"] == "offline; no robot connection"
    assert result["provenance"]["calibration"] == calibration
    with pytest.raises(FileExistsError):
        trajectory.main()
