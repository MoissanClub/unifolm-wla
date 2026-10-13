"""Record aligned stationary-arm demonstrations and convert them to LeRobot v3.

DemoWriter is called inside the teleoperation loop with MEASURED observations
and COMMANDED arm/hand targets. Prediction logs alone are not demonstrations.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import FPS, array, validate_observation
from .robot import ARM_IDS
from .semantics import native_hand_to_policy, pelvis_pose_to_world

STATE_KEYS = {
    "left_ee_pose": "left_ee_pose_gripper_base", "right_ee_pose": "right_ee_pose_gripper_base",
    "left_fig6d": "left_fig6d", "right_fig6d": "right_fig6d", "waist_joint": "waist_state_joint",
    "left_leg": "left_leg", "right_leg": "right_leg", "base_rot": "state_base_rot",
}
ACTION_KEYS = {k: v for k, v in STATE_KEYS.items() if k in ("left_ee_pose", "right_ee_pose", "left_fig6d", "right_fig6d")}
IMAGE_KEYS = {"head_left": "head_stereo_left", "cam_wrist_left": "wrist_left", "cam_wrist_right": "wrist_right"}


class DemoWriter:
    def __init__(self, directory, kinematics):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.kin = kinematics
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = deque()
        self.peak_pending_writes = 0
        self.index = 0
        self.error = None

    def add(self, observation, arm_targets, hand_targets):
        validate_observation(observation, allow_missing_cameras=True)
        if self.error:
            raise RuntimeError(self.error)
        while self.pending and self.pending[0].done():
            try:
                self.pending.popleft().result()
            except Exception as exc:
                self.error = f"Recorder write failed: {exc}"
                raise RuntimeError(self.error) from exc
        action = {}
        for side, ids in ARM_IDS.items():
            motors = np.asarray(observation["motors"]).copy()
            motors[list(ids)] = array(arm_targets[side], (5,), f"{side} arm command")
            pose = pelvis_pose_to_world(observation["base_pose"], self.kin.fk(motors, side))
            action[f"{side}_ee_pose"] = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")]
            hand = array(hand_targets[side], (6,), f"{side} hand command")
            action[f"{side}_fig6d"] = native_hand_to_policy(hand)
        record = {"observation": copy.deepcopy(observation), "action": action}
        self.pending.append(self.executor.submit(self._write, self.index, record))
        self.peak_pending_writes = max(self.peak_pending_writes, len(self.pending))
        self.index += 1

    def _write(self, index, record):
        from PIL import Image
        obs = record["observation"]
        images = obs.pop("images")
        obs["images"] = {}
        for role, pixels in images.items():
            path = f"{index:06d}.{role}.jpg"
            Image.fromarray(pixels).save(self.directory / path, quality=95)
            obs["images"][role] = path
        def numpy_json(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            raise TypeError(type(value).__name__)
        (self.directory / f"{index:06d}.json").write_text(json.dumps(record, default=numpy_json))

    def close(self, success):
        failure = None
        try:
            for future in self.pending:
                try:
                    future.result()
                except Exception as exc:
                    self.error = f"Recorder write failed: {exc}"
                    failure = exc
        finally:
            self.executor.shutdown(wait=True)
        (self.directory / "episode.json").write_text(json.dumps({
            "success": bool(success) and self.error is None, "frames": self.index,
            "error": self.error, "action_profile": "stationary_arms",
            "peak_pending_writes": self.peak_pending_writes,
        }, indent=2))
        if failure:
            raise RuntimeError(self.error) from failure


def resample_timestamps(stamps, fps=FPS):
    stamps = np.asarray(stamps, dtype=float)
    if stamps.ndim != 1 or not len(stamps) or not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
        raise ValueError("Need finite, strictly increasing capture timestamps")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be positive and finite")
    grid = np.arange(stamps[0], stamps[-1]+1e-7, 1/fps)
    right = np.searchsorted(stamps, grid).clip(0, len(stamps)-1)
    left = (right-1).clip(0)
    indices = np.where(np.abs(stamps[left]-grid) < np.abs(stamps[right]-grid), left, right)
    timing = {
        "capture_frames": len(stamps), "resampled_frames": len(indices),
        "max_capture_gap_s": float(np.max(np.diff(stamps), initial=0)),
        "max_resampling_error_s": float(np.max(np.abs(stamps[indices]-grid))),
        "repeated_frames": int(np.sum(np.diff(indices) == 0)),
    }
    return indices, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, help="Parent directory containing episode directories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/g1-fridge-stationary")
    parser.add_argument("--successful-only", action="store_true",
                        help="Include only episodes annotated successful")
    args = parser.parse_args()
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = None
    expected_roles = None
    conversion = []
    try:
        for episode in sorted(args.episodes.iterdir()):
            if not (episode / "episode.json").exists():
                continue
            metadata = json.loads((episode / "episode.json").read_text())
            if args.successful_only and not metadata["success"]:
                continue
            records = [json.loads(p.read_text()) for p in sorted(episode.glob("[0-9]*.json"))]
            if len(records) != metadata["frames"]:
                raise ValueError(f"Incomplete episode: {episode}")
            indices, timing = resample_timestamps([r["observation"]["captured_monotonic"] for r in records])
            conversion.append({"episode": str(episode), "success": metadata["success"], **timing})
            print(json.dumps(conversion[-1]), flush=True)
            for index in indices:
                record = records[int(index)]
                obs, action = record["observation"], record["action"]
                frame = {f"observation.state.{key}": np.asarray(obs["state"][name], dtype=np.float32)
                         for name, key in STATE_KEYS.items()}
                frame.update({f"action.{key}": np.asarray(action[name], dtype=np.float32)
                              for name, key in ACTION_KEYS.items()})
                roles = tuple(obs["images"])
                if expected_roles is not None and roles != expected_roles:
                    raise ValueError("Camera roles changed between frames/episodes")
                expected_roles = roles
                for role, file in obs["images"].items():
                    frame[f"observation.images.{IMAGE_KEYS[role]}"] = np.asarray(Image.open(episode / file).convert("RGB"))
                if dataset is None:
                    features = {k: {"dtype": "video" if k.startswith("observation.images.") else "float32",
                                    "shape": v.shape, "names": ["height", "width", "channel"]
                                    if k.startswith("observation.images.") else None}
                                for k, v in frame.items()}
                    dataset = LeRobotDataset.create(repo_id=args.repo_id, root=args.out, fps=FPS,
                                                    robot_type="unitree_g1_23_brainco", features=features,
                                                    use_videos=True, vcodec="libx264")
                frame["task"] = obs["instruction"]
                dataset.add_frame(frame)
            dataset.save_episode()
        if dataset is None:
            raise ValueError("No complete episodes matching the requested selection")
    finally:
        if dataset:
            dataset.finalize()
            (args.out / "conversion_timing.json").write_text(json.dumps(conversion, indent=2))


if __name__ == "__main__":
    main()
