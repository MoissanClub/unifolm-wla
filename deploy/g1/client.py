"""Record observations, evaluate in shadow mode, or explicitly test free-space arms."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from model_server.tools.msgpack_numpy import packb, unpackb

from .contract import PROTOCOL


def request(url, observation=None, timeout=30):
    endpoint = "/predict" if observation is not None else "/health"
    body = packb(observation) if observation is not None else None
    req = urllib.request.Request(url.rstrip("/")+endpoint, body, {"Content-Type": "application/msgpack"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = unpackb(response.read(24*1024*1024))
    except urllib.error.HTTPError as exc:
        details = unpackb(exc.read(24*1024*1024))
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
    parser.add_argument("--wrist-left", help="V4L2 device path")
    parser.add_argument("--wrist-right", help="V4L2 device path")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", default="results/g1")
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument("--replay", type=Path, help="A previously recorded observation .msgpack; never executes")
    parser.add_argument("--execute-free-space-arms", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    execute = args.execute_free_space_arms
    if execute and (args.record_only or args.replay):
        parser.error("Execution cannot be combined with recording/replay")
    calibration = json.loads(Path(args.calibration).read_text())
    if execute and calibration.get("verified") is not True:
        parser.error("Execution requires measured, verified wrist frame calibration")
    health = None if args.record_only else request(args.server)
    run = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    (run / "run.json").write_text(json.dumps({
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "calibration": calibration, "server": health,
        "mode": "free-space-arm-retargeting" if execute else "record" if args.record_only else "shadow",
    }, indent=2))
    sensors = arms = None
    latency = []
    try:
        if not args.replay:
            from .robot import Kinematics23, Sensors
            kin = Kinematics23(args.urdf, calibration)
            sensors = Sensors(args.interface, kin, args.wrist_left, args.wrist_right)
        if execute:
            from .arms import ArmSession
            # Operator must have disabled competing rt/arm_sdk publishers before this command.
            arms = ArmSession(sensors)
        with (run / "metrics.jsonl").open("w") as log:
            for step in range(args.steps):
                obs = unpackb(args.replay.read_bytes()) if args.replay else sensors.observe(args.instruction)
                obs["instruction"] = args.instruction
                (run / f"{step:05d}.observation.msgpack").write_bytes(packb(obs))
                if args.record_only:
                    continue
                start = time.monotonic()
                timeout = 30
                if arms:
                    timeout = calibration["limits"]["max_observation_age_s"] - (start-obs["captured_monotonic"])
                    if timeout <= 0:
                        raise RuntimeError("Observation already stale before policy request")
                reply = request(args.server, obs, timeout=timeout)
                elapsed = time.monotonic()-start
                latency.append(elapsed*1000)
                (run / f"{step:05d}.prediction.msgpack").write_bytes(packb(reply))
                metrics = {"step": step, "roundtrip_ms": elapsed*1000, **reply["metrics"]}
                if sensors:
                    from .guard import choose_target
                    try:
                        _, measured, _ = sensors.feedback()
                        targets, index = choose_target(reply, obs, measured, kin, calibration["limits"])
                        metrics.update(guard="accepted", action_index=index)
                        if arms:
                            arms.move(targets, calibration["limits"]["move_duration_s"])
                    except (ValueError, RuntimeError) as exc:
                        metrics.update(guard="rejected", reason=str(exc))
                        if arms:
                            log.write(json.dumps(metrics)+"\n")
                            raise
                print(json.dumps(metrics), flush=True)
                log.write(json.dumps(metrics)+"\n")
                log.flush()
    finally:
        try:
            if arms:
                arms.close()
        finally:
            if sensors:
                sensors.close()
        if latency:
            summary = {"samples": len(latency), "p50_ms": float(np.percentile(latency, 50)),
                       "p95_ms": float(np.percentile(latency, 95)), "max_ms": max(latency),
                       "note": "Includes cold first request; action agreement is not task success."}
            (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {run}")


if __name__ == "__main__":
    main()
