"""Record observations, predict, preview trajectories, or explicitly execute one arm.

Stream observation age uses local decode completion and sensor capture start;
it does not measure camera exposure time or establish sensor synchronization.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from model_server.tools.msgpack_numpy import packb, unpackb

from .contract import PROTOCOL


def request(url, observation=None, timeout=None):
    endpoint = "/predict" if observation is not None else "/health"
    body = packb(observation) if observation is not None else None
    req = urllib.request.Request(url.rstrip("/")+endpoint, body, {"Content-Type": "application/msgpack"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = unpackb(response.read())
    except urllib.error.HTTPError as exc:
        details = unpackb(exc.read())
        raise RuntimeError(f"Policy server: {details.get('error', exc.code)}") from exc
    if result.get("protocol") != PROTOCOL:
        raise ValueError("Wrong inference server protocol")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8601")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--interface", default="enP8p1s0")
    parser.add_argument("--urdf", default="../xr_teleoperate/assets/g1/g1_body23.urdf")
    parser.add_argument("--calibration", default="deploy/g1/calibration.example.json")
    parser.add_argument("--head-stream", help="Read an existing video stream URL instead of opening the head camera")
    parser.add_argument("--wrist-left", help="V4L2 device path")
    parser.add_argument("--wrist-right", help="V4L2 device path")
    parser.add_argument("--steps", type=int, help="Number of observations; omitted runs until Ctrl-C")
    parser.add_argument("--out", default="results/g1")
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument("--replay", type=Path, help="A previously recorded observation .msgpack; never executes")
    parser.add_argument("--plan", action="store_true", help="Save arm/hand trajectory and IK errors without motion")
    parser.add_argument("--execute", action="store_true", help="Execute planned arm/hand chunks using the supplied calibration")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    execute = args.execute
    if execute and (args.record_only or args.replay):
        parser.error("Execution cannot be combined with recording/replay")
    calibration = json.loads(Path(args.calibration).read_text())
    run = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    metadata = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "calibration": calibration, "server": None,
        "mode": "arm-hand-execution" if execute else "record" if args.record_only else "inference",
    }
    (run / "run.json").write_text(json.dumps(metadata, indent=2))
    sensors = arms = head_stream = None
    latency = []
    primary_error = None
    outcome = {"status": "running", "observations": 0, "predictions": 0,
               "completed_chunks": 0, "execution_requested": execute,
               "last_phase": "server", "task_success": None}
    try:
        health = None if args.record_only else request(args.server)
        metadata["server"] = health
        (run / "run.json").write_text(json.dumps(metadata, indent=2))
        if not args.replay:
            outcome["last_phase"] = "sensors"
            from .robot import Kinematics23, Sensors
            kin = Kinematics23(args.urdf, calibration)
            sensors = Sensors(args.interface, kin, args.wrist_left, args.wrist_right,
                              external_images=bool(args.head_stream))
            if args.head_stream:
                outcome["last_phase"] = "head_stream"
                from .camera import HeadStream
                head_stream = HeadStream(args.head_stream)
        with (run / "metrics.jsonl").open("w") as log:
            for step in itertools.count() if args.steps is None else range(args.steps):
                outcome.update(last_step=step, last_phase="capture")
                if args.replay:
                    obs = unpackb(args.replay.read_bytes())
                elif head_stream is not None:
                    rgb, image_metadata = head_stream.read()
                    obs = sensors.observe(args.instruction, images={"head_left": rgb})
                    obs.setdefault("image_metadata", {})["head_left"] = image_metadata
                    obs["captured_monotonic"] = min(obs["captured_monotonic"],
                                                    image_metadata["decode_monotonic"])
                else:
                    obs = sensors.observe(args.instruction)
                obs["instruction"] = args.instruction
                (run / f"{step:05d}.observation.msgpack").write_bytes(packb(obs))
                outcome["observations"] += 1
                if args.record_only:
                    continue
                outcome["last_phase"] = "inference"
                start = time.monotonic()
                reply = request(args.server, obs)
                elapsed = time.monotonic()-start
                latency.append(elapsed*1000)
                (run / f"{step:05d}.prediction.msgpack").write_bytes(packb(reply))
                outcome["predictions"] += 1
                metrics = {"step": step, "roundtrip_ms": elapsed*1000, **reply["metrics"]}
                metrics.update(mode="inference", executed=False,
                               observation_age_s=time.monotonic()-obs["captured_monotonic"]
                               if sensors else None, capture_ms=obs.get("capture_ms"),
                               observation_age_basis="local_capture_start_or_decode_completion_not_exposure")
                if "image_metadata" in obs:
                    metrics["image_metadata"] = obs["image_metadata"]
                if args.plan or execute:
                    outcome["last_phase"] = "planning"
                    from .trajectory import plan_chunk
                    if sensors is None:
                        from .robot import Kinematics23
                        kin = Kinematics23(args.urdf, calibration)
                    plan = plan_chunk(reply, obs, kin, side=args.side, fps=health["fps"])
                    (run / f"{step:05d}.plan.msgpack").write_bytes(packb(plan))
                    metrics["plan"] = {"side": args.side, "frames": len(plan["joints"]),
                                       "position_error_max_m": float(np.max(plan["errors"]["position_error_m"])),
                                       "rotation_error_max_rad": float(np.max(plan["errors"]["rotation_error_rad"]))}
                    if execute:
                        if arms is None:
                            outcome["last_phase"] = "arm_start"
                            from .arms import ArmSession
                            arms = ArmSession(sensors, side=args.side, fps=health["fps"])
                            arms.start()
                        outcome["last_phase"] = "execution"
                        arms.play(plan)
                        outcome["completed_chunks"] += 1
                        metrics.update(mode="arm-hand-execution", executed=True)
                print(json.dumps(metrics), flush=True)
                log.write(json.dumps(metrics)+"\n")
                log.flush()
        outcome["status"] = "completed"
    except BaseException as exc:
        primary_error = exc
        outcome.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cleanup_errors = []
        for name, resource in (("arms", arms), ("head_stream", head_stream), ("sensors", sensors)):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    cleanup_errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            outcome["cleanup_errors"] = cleanup_errors
            if primary_error is None:
                outcome["status"] = "failed"
        if latency:
            summary = {"samples": len(latency), "p50_ms": float(np.percentile(latency, 50)),
                       "p95_ms": float(np.percentile(latency, 95)), "max_ms": max(latency),
                       "note": "Includes cold first request; action agreement is not task success."}
            (run / "summary.json").write_text(json.dumps(summary, indent=2))
        (run / "outcome.json").write_text(json.dumps(outcome, indent=2))
        print(f"Saved {run}", flush=True)
        if cleanup_errors and primary_error is None:
            raise RuntimeError("; ".join(cleanup_errors))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
