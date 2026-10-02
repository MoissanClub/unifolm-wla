"""Compare a real-context 4-step action rollout in PyTorch and TensorRT on the target GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-abs", type=float, default=0.05, help="Normalized action tolerance, not a motion certificate")
    args = parser.parse_args()
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from unifolm_wla.model.framework.share_tools import dict_to_namespace, read_mode_config
    from unifolm_wla.model.modules.action_model.MMDiT_ActionHeader import get_action_model
    from .trt import TRTVelocity, velocity_wrapper
    cfg, _ = read_mode_config(args.checkpoint)
    with init_empty_weights():
        head = get_action_model(dict_to_namespace(cfg))
    with safe_open(args.checkpoint, framework="pt", device="cpu") as weights:
        for key in head.state_dict():
            set_module_tensor_to_device(head, key, "cuda", value=weights.get_tensor("action_model."+key),
                                        dtype=torch.bfloat16)
    head.to(device="cuda").eval().requires_grad_(False)
    reference = velocity_wrapper(head)
    engine = TRTVelocity(args.engine)
    with np.load(args.probe, allow_pickle=False) as probe:
        hidden = torch.as_tensor(probe["hidden"], device="cuda", dtype=torch.bfloat16)
        text_mask = torch.as_tensor(probe["text_mask"], device="cuda", dtype=torch.bool)
        mask = torch.as_tensor(probe["action_mask"], device="cuda", dtype=torch.float32)
    padding = engine.max_tokens-hidden.shape[1]
    if padding < 0:
        raise ValueError("Probe is longer than engine capacity")
    hidden = torch.nn.functional.pad(hidden, (0, 0, 0, padding))
    text_mask = torch.nn.functional.pad(text_mask, (0, padding), value=False)
    torch.manual_seed(42)
    action = torch.randn((1, head.action_horizon, head.action_output_dim), device="cuda")*mask[:, None]
    trt_action = action.clone()
    steps = head.num_inference_timesteps
    step_errors = []
    with torch.inference_mode():
        for i in range(steps):
            ts = torch.tensor([int(i/steps*head.num_timestep_buckets)], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                velocity = reference(hidden, text_mask, action, mask, ts)
            trt_velocity = engine(hidden=hidden, text_mask=text_mask, actions=trt_action,
                                  action_mask=mask, timestep=ts)
            action = (action+velocity.float()/steps)*mask[:, None]
            trt_action = (trt_action+trt_velocity.float()/steps)*mask[:, None]
            step_errors.append(float((action-trt_action).abs().max()))
    difference = (action-trt_action).abs()
    passed = bool(torch.isfinite(difference).all()) and float(difference.max()) <= args.max_abs
    report = {"probe": str(args.probe), "engine": args.engine, "checkpoint": args.checkpoint,
              "step_max_abs": step_errors, "rollout_max_abs": float(difference.max()),
              "rollout_mean_abs": float(difference.mean()), "tolerance": args.max_abs, "passed": passed,
              "limitation": "One probe; repeat over held-out tasks, poses, masks and camera sets before deployment."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
