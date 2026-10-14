"""Exercise CLI orchestration with fake sensors, HTTP, planning, and publishers."""
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.g1 import arms, camera, client, robot, trajectory
from deploy.g1.contract import PROTOCOL, WBTAdapter
from model_server.tools.msgpack_numpy import packb, unpackb


@pytest.fixture
def client_run(tmp_path, monkeypatch, observation, statistics, calibration):
    events, faults, calls = [], {}, {}
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    output = tmp_path / "runs"
    stream_rgb = np.array([[[17, 31, 47]]], dtype=np.uint8)
    stream_options = {"decode_monotonic": 90.0}

    def visit(stage):
        events.append(stage)
        calls[stage] = calls.get(stage, 0) + 1
        failure = faults.get((stage, calls[stage]))
        if failure is not None:
            raise failure

    class Kinematics:
        def __init__(self, *args):
            visit("kinematics")

    class Sensors:
        def __init__(self, *args, external_images=False):
            visit("sensors:init")
            self.external_images = external_images
            if not external_images:
                visit("sensors:direct_camera")

        def observe(self, instruction, images=None):
            visit("observe")
            obs = copy.deepcopy(observation)
            obs["instruction"] = instruction
            if self.external_images:
                assert images is not None and "head_left" in images
                obs["images"] = images
            else:
                assert images is None
            return obs

        def close(self):
            visit("sensors:close")

    class HeadStream:
        def __init__(self, url):
            visit("stream:init")
            self.url = url

        def read(self):
            visit("stream:read")
            return stream_rgb.copy(), {
                "source": self.url, "decode_monotonic": stream_options["decode_monotonic"],
                "sequence": calls["stream:read"],
                "timestamp_kind": "local_decode_completion_not_exposure",
            }

        def close(self):
            visit("stream:close")

    class ArmSession:
        def __init__(self, sensors, side, fps):
            visit("arms:init")
            assert isinstance(sensors, Sensors)
            assert side == "left"
            assert fps == 30

        def start(self):
            visit("arms:start")

        def play(self, plan):
            visit("arms:play")
            assert plan["joints"].shape == (30, 5)

        def close(self):
            visit("arms:close")

    def request(url, obs=None, **kwargs):
        if obs is None:
            visit("health")
            return {"protocol": PROTOCOL, "fps": 30, "horizon": 30}
        visit("predict")
        reply = WBTAdapter(statistics).decode(np.zeros((30, 54), dtype=np.float32), obs)
        return {**reply, "protocol": PROTOCOL, "frame_id": obs["frame_id"],
                "metrics": {"total_ms": 1500.0}}

    def plan_chunk(reply, obs, kin, side, fps):
        visit("plan")
        assert reply["frame_id"] == obs["frame_id"]
        assert isinstance(kin, Kinematics)
        return {"side": side, "fps": fps, "joints": np.zeros((30, 5)),
                "hands": np.zeros((30, 6)),
                "errors": {"position_error_m": np.zeros(30),
                           "rotation_error_rad": np.zeros(30)}}

    monkeypatch.setattr(robot, "Kinematics23", Kinematics)
    monkeypatch.setattr(robot, "Sensors", Sensors)
    monkeypatch.setattr(camera, "HeadStream", HeadStream)
    monkeypatch.setattr(arms, "ArmSession", ArmSession)
    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(trajectory, "plan_chunk", plan_chunk)

    def run(*args):
        monkeypatch.setattr(sys, "argv", [
            "deploy.g1.client", "--steps", "1", "--out", str(output),
            "--calibration", str(calibration_path),
            "--instruction", "pick up the can with the left hand", *map(str, args),
        ])
        client.main()

    return SimpleNamespace(run=run, events=events, faults=faults, output=output,
                           observation=observation, tmp_path=tmp_path,
                           stream_rgb=stream_rgb, stream_options=stream_options)


@pytest.mark.parametrize("preview", [False, True])
def test_inference_and_preview_never_construct_arm_session(client_run, preview):
    client_run.run(*(["--plan"] if preview else []))
    assert not any(event.startswith("arms:") for event in client_run.events)
    assert ("plan" in client_run.events) is preview
    assert client_run.events[-1] == "sensors:close"
    run_dir, = client_run.output.iterdir()
    metric = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert metric["executed"] is False
    assert (run_dir / "00000.prediction.msgpack").exists()
    assert (run_dir / "00000.plan.msgpack").exists() is preview
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["status"] == "completed"
    assert outcome["observations"] == outcome["predictions"] == 1
    assert outcome["completed_chunks"] == 0
    assert outcome["execution_requested"] is False
    assert outcome["task_success"] is None


def test_execution_plans_before_acquiring_arm_authority(client_run):
    client_run.run("--execute", "--steps", "2")
    events = client_run.events
    assert events.index("predict") < events.index("plan") < events.index("arms:init")
    assert events.index("arms:init") < events.index("arms:start") < events.index("arms:play")
    assert events.count("arms:start") == 1
    assert events.count("arms:play") == 2
    assert events[-2:] == ["arms:close", "sensors:close"]
    run_dir, = client_run.output.iterdir()
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [metric["executed"] for metric in metrics] == [True, True]
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["status"] == "completed"
    assert outcome["observations"] == outcome["predictions"] == outcome["completed_chunks"] == 2
    assert outcome["execution_requested"] is True
    assert outcome["task_success"] is None


@pytest.mark.parametrize("stage", ["predict", "plan"])
@pytest.mark.parametrize("failure_step", [1, 2])
def test_prediction_or_plan_failure_releases_existing_resources(client_run, stage, failure_step):
    client_run.faults[(stage, failure_step)] = RuntimeError(f"{stage} failed")
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        client_run.run("--execute", "--steps", "2")
    assert client_run.events[-1] == "sensors:close"
    if failure_step == 1:
        assert not any(event.startswith("arms:") for event in client_run.events)
    else:
        assert client_run.events[-2:] == ["arms:close", "sensors:close"]
        assert client_run.events.count("arms:play") == 1
    run_dir, = client_run.output.iterdir()
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["status"] == "failed"
    assert outcome["observations"] == failure_step
    assert outcome["predictions"] == failure_step - (stage == "predict")
    assert outcome["completed_chunks"] == failure_step - 1
    assert outcome["last_phase"] == {"predict": "inference", "plan": "planning"}[stage]
    assert outcome["error"] == f"RuntimeError: {stage} failed"
    assert outcome["task_success"] is None


@pytest.mark.parametrize("stage", ["arms:start", "arms:play"])
def test_arm_failure_releases_arm_and_sensor_resources(client_run, stage):
    client_run.faults[(stage, 1)] = RuntimeError("controller unavailable")
    with pytest.raises(RuntimeError, match="controller unavailable"):
        client_run.run("--execute")
    assert client_run.events[-2:] == ["arms:close", "sensors:close"]
    run_dir, = client_run.output.iterdir()
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["status"] == "failed"
    assert outcome["completed_chunks"] == 0
    assert outcome["last_phase"] == {"arms:start": "arm_start", "arms:play": "execution"}[stage]


def test_interrupt_during_execution_releases_resources(client_run):
    client_run.faults[("arms:play", 1)] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        client_run.run("--execute")
    assert client_run.events[-2:] == ["arms:close", "sensors:close"]
    run_dir, = client_run.output.iterdir()
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["status"] == "interrupted"
    assert outcome["completed_chunks"] == 0
    assert outcome["task_success"] is None


def test_cleanup_failure_preserves_primary_execution_error(client_run):
    client_run.faults[("arms:play", 1)] = RuntimeError("trajectory failed")
    client_run.faults[("arms:close", 1)] = OSError("release failed")
    client_run.faults[("sensors:close", 1)] = OSError("camera release failed")
    with pytest.raises(RuntimeError, match="trajectory failed"):
        client_run.run("--execute")
    assert client_run.events[-2:] == ["arms:close", "sensors:close"]
    run_dir, = client_run.output.iterdir()
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["error"] == "RuntimeError: trajectory failed"
    assert outcome["cleanup_errors"] == [
        "arms: OSError: release failed", "sensors: OSError: camera release failed",
    ]
    assert (run_dir / "summary.json").exists()


def test_cleanup_failure_after_completed_chunk_reports_failure(client_run):
    client_run.faults[("arms:close", 1)] = OSError("release failed")
    with pytest.raises(RuntimeError, match="release failed"):
        client_run.run("--execute")
    assert client_run.events[-2:] == ["arms:close", "sensors:close"]
    run_dir, = client_run.output.iterdir()
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["status"] == "failed"
    assert outcome["completed_chunks"] == 1
    assert outcome["task_success"] is None
    assert outcome["cleanup_errors"] == ["arms: OSError: release failed"]


def test_unavailable_server_saves_failure_without_initializing_hardware(client_run, capsys):
    client_run.faults[("health", 1)] = ConnectionError("server stopped")
    with pytest.raises(ConnectionError, match="server stopped"):
        client_run.run("--execute")
    assert client_run.events == ["health"]
    run_dir, = client_run.output.iterdir()
    outcome = json.loads((run_dir / "outcome.json").read_text())
    metadata = json.loads((run_dir / "run.json").read_text())
    assert outcome["status"] == "failed"
    assert outcome["last_phase"] == "server"
    assert outcome["observations"] == outcome["predictions"] == outcome["completed_chunks"] == 0
    assert metadata["server"] is None
    assert f"Saved {run_dir}" in capsys.readouterr().out


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_replay_never_initializes_sensors_or_arm_publishers(client_run, preview, stream):
    path = client_run.tmp_path / "replay.msgpack"
    path.write_bytes(packb(client_run.observation))
    stream_args = ["--head-stream", "rtsp://127.0.0.1:8554/d435"] if stream else []
    client_run.run("--replay", path, *(["--plan"] if preview else []), *stream_args)
    assert not any(event.startswith(("sensors:", "arms:", "stream:")) for event in client_run.events)
    assert ("kinematics" in client_run.events) is preview
    run_dir, = client_run.output.iterdir()
    obs = unpackb((run_dir / "00000.observation.msgpack").read_bytes())
    assert obs["instruction"] == "pick up the can with the left hand"


@pytest.mark.parametrize("mode", ["--replay", "--record-only"])
def test_execution_rejects_read_only_modes_before_any_io(client_run, mode):
    args = ["--execute", mode]
    if mode == "--replay":
        args.append(str(client_run.tmp_path / "does-not-need-to-exist.msgpack"))
    with pytest.raises(SystemExit) as error:
        client_run.run(*args)
    assert error.value.code == 2
    assert client_run.events == []
    assert not client_run.output.exists()


def test_record_only_does_not_call_policy_or_construct_arm_session(client_run):
    client_run.run("--record-only")
    assert "health" not in client_run.events
    assert "predict" not in client_run.events
    assert not any(event.startswith("arms:") for event in client_run.events)
    assert client_run.events[-1] == "sensors:close"
    run_dir, = client_run.output.iterdir()
    assert (run_dir / "00000.observation.msgpack").exists()
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["observations"] == 1
    assert outcome["predictions"] == outcome["completed_chunks"] == 0
    assert outcome["status"] == "completed"


@pytest.mark.parametrize("record_only", [False, True])
@pytest.mark.parametrize("decoded", [90.0, 110.0])
def test_head_stream_supplies_rgb_without_opening_camera(client_run, record_only, decoded):
    url = "rtsp://127.0.0.1:8554/d435"
    client_run.stream_options["decode_monotonic"] = decoded
    client_run.run("--head-stream", url, *(["--record-only"] if record_only else []))
    assert "sensors:direct_camera" not in client_run.events
    assert not any(event.startswith("arms:") for event in client_run.events)
    assert client_run.events[-2:] == ["stream:close", "sensors:close"]
    run_dir, = client_run.output.iterdir()
    obs = unpackb((run_dir / "00000.observation.msgpack").read_bytes())
    np.testing.assert_array_equal(obs["images"]["head_left"], client_run.stream_rgb)
    assert obs["captured_monotonic"] == min(100.0, decoded)
    assert obs["image_metadata"]["head_left"] == {
        "source": url, "decode_monotonic": decoded, "sequence": 1,
        "timestamp_kind": "local_decode_completion_not_exposure",
    }
    if record_only:
        assert "health" not in client_run.events
        assert "predict" not in client_run.events
    else:
        metrics = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
        assert metrics["observation_age_basis"] == "local_capture_start_or_decode_completion_not_exposure"
        assert metrics["image_metadata"] == obs["image_metadata"]


def test_head_stream_read_failure_releases_stream_and_sensors(client_run):
    client_run.faults[("stream:read", 1)] = RuntimeError("decode failed")
    with pytest.raises(RuntimeError, match="decode failed"):
        client_run.run("--head-stream", "rtsp://127.0.0.1:8554/d435", "--execute")
    assert not any(event.startswith("arms:") for event in client_run.events)
    assert client_run.events[-2:] == ["stream:close", "sensors:close"]
    run_dir, = client_run.output.iterdir()
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["status"] == "failed"
    assert outcome["last_phase"] == "capture"
    assert outcome["observations"] == outcome["predictions"] == 0


def test_execution_closes_arms_before_stream_and_sensor_resources(client_run):
    client_run.run("--head-stream", "rtsp://127.0.0.1:8554/d435", "--execute")
    assert client_run.events[-3:] == ["arms:close", "stream:close", "sensors:close"]


def test_stream_cleanup_failure_preserves_primary_error_and_closes_sensors(client_run):
    client_run.faults[("stream:read", 1)] = RuntimeError("decode failed")
    client_run.faults[("stream:close", 1)] = OSError("stream release failed")
    with pytest.raises(RuntimeError, match="decode failed"):
        client_run.run("--head-stream", "rtsp://127.0.0.1:8554/d435")
    assert client_run.events[-2:] == ["stream:close", "sensors:close"]
    run_dir, = client_run.output.iterdir()
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["error"] == "RuntimeError: decode failed"
    assert outcome["cleanup_errors"] == ["head_stream: OSError: stream release failed"]
