"""Offline episode evaluation with explicit normalization and WBT metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import SLICES


def action_errors(pred, truth, mask):
    result = {"normalized_mae": float(np.abs(pred-truth)[:, mask].mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--unnorm-key", default="UnifoLM_WBT")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("results/g1-offline"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.stride < 1 or args.max_frames < 1:
        parser.error("Positive stride and frame count required")
    from .inference import Policy
    from unifolm_wla.dataloader.multi_source_dataset.config import load_config
    from unifolm_wla.dataloader.multi_source_dataset.single_source_dataset import create_single_source_dataset
    from examples.unifolm_wla.eval_files.unitree.eval_local_episode import (
        _episode_frame_range, _build_example_from_sample,
    )
    config = load_config(args.data_config)
    enabled = [s for s in config.datasets if s.enabled]
    if len(enabled) != 1:
        raise ValueError("Evaluation config must select exactly one source")
    source = create_single_source_dataset(enabled[0], config)
    policy = Policy(args.checkpoint, image_size=config.image_size)
    # Never silently use Dex1 stats to decode WBT actions.
    stats = policy.model.norm_stats[args.unnorm_key]
    for kind in ("action", "state"):
        for field in ("offset", "scale"):
            expected = np.asarray(stats[kind][field])
            actual = getattr(source, f"_{kind}_norm_{field}")
            if not np.allclose(actual, expected, atol=1e-5):
                raise ValueError(f"Dataset/checkpoint {kind}.{field} differs; reconcile normalization first")
    task_offset, start, end = _episode_frame_range(source, args.task, args.episode)
    end = min(end, start+args.max_frames)
    scale = np.asarray(stats["action"]["scale"])
    offset = np.asarray(stats["action"]["offset"])
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    with (args.out / "metrics.jsonl").open("w") as log:
        for frame in range(start, end, args.stride):
            sample = source[task_offset+frame]
            example = _build_example_from_sample(sample)
            pred, timing = policy.predict(example, seed=args.seed+frame)
            truth = sample["action"].numpy()
            # Exclude padded tail frames at episode boundaries from metrics.
            n = min(len(pred), len(truth), end-frame)
            pred, truth = pred[:n], truth[:n]
            mask = sample["action_mask"].numpy().astype(bool)
            metric = {"frame": frame, **timing, **action_errors(pred, truth, mask)}
            p, t = pred*scale+offset, truth*scale+offset
            for side in ("left", "right"):
                slot = SLICES[f"{side}_xyz_rotvec"]
                pp, tt = p[:, slot], t[:, slot]
                metric[f"{side}_position_mae_m"] = float(np.linalg.norm(pp[:, :3]-tt[:, :3], axis=1).mean())
                rp, rt = Rotation.from_rotvec(pp[:, 3:]), Rotation.from_rotvec(tt[:, 3:])
                metric[f"{side}_rotation_mae_rad"] = float((rp.inv()*rt).magnitude().mean())
                hand = SLICES[f"{side}_fig6d"]
                metric[f"{side}_finger_mae"] = float(np.abs(p[:, hand]-t[:, hand]).mean())
            np.savez_compressed(args.out / f"{frame:06d}.npz", predicted=pred, target=truth, mask=mask)
            records.append(metric)
            log.write(json.dumps(metric)+"\n")
            log.flush()
    times = [r["total_ms"] for r in records]
    summary = {"chunks": len(records), "p50_ms": float(np.percentile(times, 50)),
               "p95_ms": float(np.percentile(times, 95)),
               "normalized_mae": float(np.mean([r["normalized_mae"] for r in records])),
               "interpretation": "Teacher-forced action agreement; not closed-loop success or novel-task generalization."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
