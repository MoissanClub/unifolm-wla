"""Benchmark a saved observation and optionally save real VLM features for TRT parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--engine")
    parser.add_argument("--save-probe", action="store_true")
    parser.add_argument("--profile", choices=("wbt", "stationary_arms"), default="wbt")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("Positive iteration count required")
    from model_server.tools.msgpack_numpy import unpackb
    from .contract import WBTAdapter
    from .inference import Policy
    policy = Policy(args.checkpoint, engine=args.engine)
    obs = unpackb(args.observation.read_bytes())
    example = WBTAdapter(policy.model.norm_stats, allow_missing_cameras=True, profile=args.profile).example(obs)
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for i in range(args.iterations+1):
        _, metrics = policy.predict(example, seed=42+i)
        records.append(metrics)
        print(json.dumps({"iteration": i, **metrics}), flush=True)
    warm = records[1:]
    summary = {"cold": records[0], "warm": warm,
               "warm_p50_ms": float(np.percentile([r["total_ms"] for r in warm], 50)),
               "warm_p95_ms": float(np.percentile([r["total_ms"] for r in warm], 95))}
    (args.out / "timings.json").write_text(json.dumps(summary, indent=2))
    if args.save_probe:
        with policy.torch.inference_mode():
            hidden, mask = policy.encode(example)
        np.savez_compressed(args.out / "probe.npz", hidden=hidden.float().cpu().numpy(),
                            text_mask=mask.cpu().numpy(), action_mask=example["action_mask"][None])


if __name__ == "__main__":
    main()
