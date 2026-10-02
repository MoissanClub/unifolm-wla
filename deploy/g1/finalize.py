"""Package a FULL fine-tuned checkpoint; merge injected LoRA layers if present.

Use the full training checkpoint, not adapter-only: action encoders/decoders
outside the LoRA-injected DiT backbone may also have trained.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def merge_lora_layers(model):
    from peft.tuners.lora.layer import LoraLayer
    for name, module in list(model.named_modules()):
        if isinstance(module, LoraLayer):
            module.merge(safe_merge=True)
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, module.get_base_layer())
    if any("lora_" in key or ".base_layer." in key for key in model.state_dict()):
        raise ValueError("Unmerged adapter weights remain")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Full steps_* or final_model/model.safetensors")
    parser.add_argument("--run-dir", required=True, type=Path, help="Contains config.yaml and dataset_statistics.json")
    parser.add_argument("--base-dir", required=True, type=Path, help="Released model bundle, supplies tokenizer")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from safetensors.torch import save_file
    from unifolm_wla.model.framework.base_framework import build_framework
    from unifolm_wla.training.trainer_utils.trainer_tools import TrainerUtils
    config_path = args.run_dir / "config.full.yaml"
    if not config_path.exists():
        config_path = args.run_dir / "config.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.framework.qwenvl.base_vlm = str((args.base_dir / "tokenizer").resolve())
    cfg.framework.qwenvl.attn_implementation = "sdpa"
    cfg.framework.robot_state_projector.load_from = None
    cfg.trainer.pretrained_checkpoint = None
    with init_empty_weights():
        model = build_framework(cfg)
        model = TrainerUtils.apply_lora_adapters(model, cfg.trainer.get("lora"))
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as weights:
        if set(weights.keys()) != set(model.state_dict()):
            raise ValueError("Full checkpoint/config mismatch. Adapter-only files cannot be finalized this way.")
        for key in weights.keys():
            tensor = weights.get_tensor(key)
            set_module_tensor_to_device(model, key, "cpu", value=tensor, dtype=tensor.dtype)
    if cfg.trainer.get("lora", {}).get("enabled", False):
        model = merge_lora_layers(model)
    model.eval().to(dtype=torch.bfloat16)
    (args.out / "checkpoints").mkdir(parents=True)
    # clone breaks any tied storage; the deployment loader verifies all expected keys.
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()},
              str(args.out / "checkpoints/model.safetensors"))
    cfg.framework.qwenvl.base_vlm = "./tokenizer"
    cfg.framework.robot_state_projector.load_from = None
    cfg.trainer = {}
    OmegaConf.save(cfg, args.out / "config.yaml")
    shutil.copyfile(args.run_dir / "dataset_statistics.json", args.out / "dataset_statistics.json")
    shutil.copytree(args.base_dir / "tokenizer", args.out / "tokenizer")
    print(args.out)


if __name__ == "__main__":
    main()
