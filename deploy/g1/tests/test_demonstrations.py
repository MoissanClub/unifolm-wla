import json
import numpy as np
import pytest

from deploy.g1.demonstrations import DemoWriter, regular_indices


def test_resample_timestamp_jitter():
    stamps = np.arange(60)/30 + np.sin(np.arange(60))*0.003
    indices = regular_indices(stamps)
    assert len(indices) >= 59
    assert np.all(np.diff(indices) > 0)


def test_recording_gap_is_not_silently_padded():
    stamps = np.arange(60)/30
    stamps[30:] += 0.2
    with pytest.raises(ValueError, match="gap"):
        regular_indices(stamps)


def test_recorder_uses_command_not_observed_target(tmp_path, observation):
    class Kin:
        def fk(self, motors, side):
            pose = np.eye(4)
            pose[:3, 3] = [motors[15 if side == "left" else 22], 0, 0]
            return pose
    writer = DemoWriter(tmp_path / "episode", Kin())
    writer.add(observation, {"left": [.2, 0, 0, 0, 0], "right": [.4, 0, 0, 0, 0]},
               {"left": [.1]*6, "right": [.8]*6})
    writer.close(success=True)
    record = json.loads((tmp_path / "episode/000000.json").read_text())
    assert record["action"]["left_ee_pose"][0] == pytest.approx(.2)
    assert record["observation"]["state"]["left_ee_pose"][0] == pytest.approx(.3)
    assert record["action"]["right_fig6d"] == pytest.approx([.8]*6)
    assert "images" in observation  # caller data was not mutated by worker
    assert json.loads((tmp_path / "episode/episode.json").read_text())["success"]


def test_recorder_rejects_hand_units(tmp_path, observation):
    class Kin:
        def fk(self, motors, side):
            return np.eye(4)
    writer = DemoWriter(tmp_path / "episode", Kin())
    try:
        with pytest.raises(ValueError, match="normalized"):
            writer.add(observation, {"left": [0]*5, "right": [0]*5}, {"left": [1000]*6, "right": [0]*6})
    finally:
        writer.close(success=False)


def test_failed_write_never_marks_episode_successful(tmp_path, observation, monkeypatch):
    class Kin:
        def fk(self, motors, side):
            return np.eye(4)
    writer = DemoWriter(tmp_path / "episode", Kin())
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(writer, "_write", fail)
    writer.add(observation, {"left": [0]*5, "right": [0]*5}, {"left": [0]*6, "right": [0]*6})
    with pytest.raises(RuntimeError, match="disk full"):
        writer.close(success=True)
    metadata = json.loads((tmp_path / "episode/episode.json").read_text())
    assert metadata["success"] is False
