import json
import threading
import numpy as np
import pytest

from deploy.g1.demonstrations import DemoWriter, resample_timestamps


def test_resample_timestamp_jitter():
    stamps = np.arange(60)/30 + np.sin(np.arange(60))*0.003
    indices, timing = resample_timestamps(stamps)
    assert len(indices) >= 59
    assert np.all(np.diff(indices) > 0)
    assert timing["max_resampling_error_s"] < .01


def test_recording_gap_is_reported_without_rejecting_episode():
    stamps = np.arange(60)/30
    stamps[30:] += 0.2
    indices, timing = resample_timestamps(stamps)
    assert len(indices) > len(stamps)
    assert timing["max_capture_gap_s"] == pytest.approx(1/30 + .2)
    assert timing["max_resampling_error_s"] > .02
    assert timing["repeated_frames"] > 0


def test_short_episode_can_be_resampled():
    indices, timing = resample_timestamps([1.0])
    np.testing.assert_array_equal(indices, [0])
    assert timing["max_capture_gap_s"] == 0


@pytest.mark.parametrize("stamps", [[], [1, 1], [2, 1], [1, np.nan]])
def test_corrupt_timestamps_are_rejected(stamps):
    with pytest.raises(ValueError, match="timestamps"):
        resample_timestamps(stamps)


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
    assert record["action"]["left_ee_pose"][0] == pytest.approx(10.2)
    assert record["observation"]["state"]["left_ee_pose"][0] == pytest.approx(.3)
    assert record["action"]["right_fig6d"] == pytest.approx([.2]*6)
    assert "images" in observation  # caller data was not mutated by worker
    assert json.loads((tmp_path / "episode/episode.json").read_text())["success"]


def test_recorder_rejects_nonfinite_hand_commands(tmp_path, observation):
    class Kin:
        def fk(self, motors, side):
            return np.eye(4)
    writer = DemoWriter(tmp_path / "episode", Kin())
    try:
        with pytest.raises(ValueError, match="finite"):
            writer.add(observation, {"left": [0]*5, "right": [0]*5}, {"left": [np.nan]*6, "right": [0]*6})
    finally:
        writer.close(success=False)


def test_recorder_keeps_frames_when_writer_falls_behind(tmp_path, observation, monkeypatch):
    class Kin:
        def fk(self, motors, side):
            return np.eye(4)
    writer = DemoWriter(tmp_path / "episode", Kin())
    release = threading.Event()
    write = writer._write
    def delayed_write(*args):
        release.wait()
        write(*args)
    monkeypatch.setattr(writer, "_write", delayed_write)
    try:
        for _ in range(12):
            writer.add(observation, {"left": [0]*5, "right": [0]*5},
                       {"left": [0]*6, "right": [0]*6})
        assert writer.peak_pending_writes == 12
    finally:
        release.set()
        writer.close(success=True)
    metadata = json.loads((tmp_path / "episode/episode.json").read_text())
    assert metadata["frames"] == 12
    assert metadata["success"]


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
