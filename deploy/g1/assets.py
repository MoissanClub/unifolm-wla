"""Pin, fetch, and inspect the released model without executing robot code."""
from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import struct
from pathlib import Path

MODEL_ID = "unitreerobotics/UnifoLM-WLA-1.0-Base"
MODEL_REVISION = "dedc0612c4a921446db3a8cb0864bb7f7433329e"
CODE_REVISION = "0a1aa87be8f775c7883fcd27216e3666ac18a8c5"


def inspect_checkpoint(path):
    with Path(path).open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 10*1024*1024:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(stream.read(length))
    result = {}
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        group = result.setdefault(key.split(".")[0], {"parameters": 0, "bytes": 0, "dtypes": []})
        group["parameters"] += math.prod(spec["shape"])
        group["bytes"] += spec["data_offsets"][1]-spec["data_offsets"][0]
        if spec["dtype"] not in group["dtypes"]:
            group["dtypes"].append(spec["dtype"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "inspect", "preflight"))
    parser.add_argument("--model-dir", type=Path, default=Path("playground/Pretrained_models/UnifoLM-WLA-1.0-Base"))
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.command == "fetch":
        from huggingface_hub import snapshot_download
        args.model_dir.mkdir(parents=True, exist_ok=True)
        if not args.metadata_only and shutil.disk_usage(args.model_dir).free < 15*2**30:
            raise RuntimeError("Need at least 15 GiB free for the base checkpoint")
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=args.model_dir,
                          allow_patterns=["config.yaml", "dataset_statistics.json", "tokenizer/*"]
                          if args.metadata_only else None)
        (args.model_dir / "download_manifest.json").write_text(json.dumps({
            "repo": MODEL_ID, "revision": MODEL_REVISION, "code_revision": CODE_REVISION,
        }, indent=2))
    elif args.command == "inspect":
        print(json.dumps(inspect_checkpoint(args.model_dir / "checkpoints/model.safetensors"), indent=2))
        print("Statistics keys:", list(json.loads((args.model_dir / "dataset_statistics.json").read_text())))
    else:
        report = {"machine": platform.machine(), "python": platform.python_version()}
        for name, path in (("l4t", "/etc/nv_tegra_release"), ("memory", "/proc/meminfo")):
            if Path(path).exists():
                lines = Path(path).read_text().splitlines()
                report[name] = lines[:3] if name == "memory" else lines[:1]
        try:
            import torch
            report.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
            if torch.cuda.is_available():
                # A working tensor kernel matters more than an installed package name.
                value = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
                report["bf16_matmul"] = float((value @ value)[0, 0])
                report["gpu_memory_free_total"] = torch.cuda.mem_get_info()
        except Exception as exc:
            report["torch_error"] = str(exc)
        try:
            import tensorrt
            report["tensorrt"] = tensorrt.__version__
        except ImportError:
            report["tensorrt"] = "not importable in this Python (may still be installed system-wide)"
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
