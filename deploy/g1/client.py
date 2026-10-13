"""Record observations, predict, preview trajectories, or explicitly execute one arm."""
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
    health = None if args.record_only else request(args.server)
    run = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    (run / "run.json").write_text(json.dumps({
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "calibration": calibration, "server": health,
        "mode": "arm-hand-execution" if execute else "record" if args.record_only else "inference",
    }, indent=2))
    sensors = arms = None
    latency = []
    try:
        if not args.replay:
            from .robot import Kinematics23, Sensors
            kin = Kinematics23(args.urdf, calibration)
            sensors = Sensors(args.interface, kin, args.wrist_left, args.wrist_right)
        with (run / "metrics.jsonl").open("w") as log:
            for step in itertools.count() if args.steps is None else range(args.steps):
                obs = unpackb(args.replay.read_bytes()) if args.replay else sensors.observe(args.instruction)
                obs["instruction"] = args.instruction
                (run / f"{step:05d}.observation.msgpack").write_bytes(packb(obs))
                if args.record_only:
                    continue
                start = time.monotonic()
                reply = request(args.server, obs)
                elapsed = time.monotonic()-start
                latency.append(elapsed*1000)
                (run / f"{step:05d}.prediction.msgpack").write_bytes(packb(reply))
                metrics = {"step": step, "roundtrip_ms": elapsed*1000, **reply["metrics"]}
                metrics.update(mode="inference", executed=False,
                               observation_age_s=time.monotonic()-obs["captured_monotonic"]
                               if sensors else None, capture_ms=obs.get("capture_ms"))
                if args.plan or execute:
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
                            from .arms import ArmSession
                            arms = ArmSession(sensors, side=args.side, fps=health["fps"])
                            arms.start()
                        arms.play(plan)
                        metrics.update(mode="arm-hand-execution", executed=True)
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
