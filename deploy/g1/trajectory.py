"""Plan a complete stationary arm/hand chunk without publishing robot commands.

The robot holds its pose during inference. Playback starts at chunk index zero
when the result arrives; inference latency does not consume this playback clock.
This is stop-and-act retargeting, not reproduction of the WBT body controller.
"""
from __future__ import annotations

import numpy as np

from .contract import FPS, PROTOCOL, array
from .robot import ARM_IDS
from .semantics import policy_hand_to_native, world_pose_to_pelvis


def plan_chunk(reply, observation, kin, side="left", fps=FPS):
    """Return all joint/hand targets plus IK and unsupported-body diagnostics."""
    if side not in ARM_IDS:
        raise ValueError(f"Unknown arm side: {side}")
    if (not isinstance(observation.get("frame_id"), str)
            or observation.get("protocol") != PROTOCOL or reply.get("protocol") != PROTOCOL
            or reply.get("frame_id") != observation.get("frame_id")):
        raise ValueError("Action belongs to a different observation/protocol")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Action frame rate must be positive and finite")
    raw = np.asarray(reply["raw"])
    if raw.ndim != 2 or raw.shape[1] != 54 or len(raw) == 0 or not np.isfinite(raw).all():
        raise ValueError("Expected a finite nonempty action chunk [T,54]")
    horizon = len(raw)
    poses = {arm: array(reply[f"{arm}_T"], (horizon, 4, 4), f"{arm}_T")
             for arm in ARM_IDS}
    hands = {arm: array(reply[f"{arm}_hand"], (horizon, 6), f"{arm}_hand")
             for arm in ARM_IDS}
    base = array(reply["base_command"], (horizon, 4), "base_command")
    relative = array(reply["base_relative"], (horizon, 6), "base_relative")
    waist = array(reply["waist"], (horizon, 3), "waist")
    legs = array(reply["legs"], (horizon, 12), "legs")
    motors = np.asarray(observation["motors"], dtype=np.float64).copy()
    if motors.ndim != 1 or len(motors) <= max(ARM_IDS["right"]) or not np.isfinite(motors).all():
        raise ValueError("Expected finite motor feedback containing both arms")
    base_pose = array(observation["base_pose"], (7,), "base_pose")
    targets = world_pose_to_pelvis(base_pose, poses[side])
    ids = list(ARM_IDS[side])
    joints, full_motors, errors = [], [], []
    for target in targets:
        q, error = kin.fit(motors, side, target)
        q = array(q, (len(ids),), f"{side} IK joints")
        motors[ids] = q
        joints.append(q.copy())
        full_motors.append(motors.copy())
        errors.append(error)
    diagnostics = {
        "execution": "stationary single-arm stop-and-act; other body outputs are not commanded",
        "base_command_max_abs": np.max(np.abs(base), axis=0).tolist(),
        "base_translation_max_m": float(np.max(np.linalg.norm(relative[:, :3], axis=1))),
        "base_rotation_max_rad": float(np.max(np.linalg.norm(relative[:, 3:], axis=1))),
        "waist_change_max_rad": float(np.max(np.abs(waist - [motors[12], 0, 0]))),
        "leg_change_max_rad": float(np.max(np.abs(legs - motors[:12]))),
        "other_arm": "right" if side == "left" else "left",
    }
    return {
        "side": side, "fps": fps, "joints": np.stack(joints),
        "hands": policy_hand_to_native(hands[side]), "motors": np.stack(full_motors),
        "time_s": np.arange(1, horizon + 1, dtype=np.float64) / fps,
        "errors": {
            "position_error_m": np.array([e["position_error_m"] for e in errors]),
            "rotation_error_rad": np.array([e["rotation_error_rad"] for e in errors]),
            "converged": np.array([e["converged"] for e in errors], dtype=bool),
        },
        "diagnostics": diagnostics,
    }


def main():
    """Rebuild a saved trajectory without an inference server or robot connection."""
    import argparse
    import json
    from pathlib import Path
    from model_server.tools.msgpack_numpy import packb, unpackb
    from .robot import Kinematics23

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--fps", type=float, default=FPS, help="WBT action frame rate")
    args = parser.parse_args()
    observation = unpackb(args.observation.read_bytes())
    prediction = unpackb(args.prediction.read_bytes())
    calibration = json.loads(args.calibration.read_text())
    plan = plan_chunk(prediction, observation, Kinematics23(args.urdf, calibration),
                      side=args.side, fps=args.fps)
    plan["provenance"] = {"observation": str(args.observation), "prediction": str(args.prediction),
                          "urdf": str(args.urdf), "calibration": calibration,
                          "execution": "offline; no robot connection"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as output:
        output.write(packb(plan))
    print(json.dumps({"saved": str(args.out), "frames": len(plan["joints"]),
                      "position_error_max_m": float(np.max(plan["errors"]["position_error_m"])),
                      "rotation_error_max_rad": float(np.max(plan["errors"]["rotation_error_rad"])),
                      **plan["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
