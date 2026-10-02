"""Generate a data config from upstream's canonical WBT schema."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import yaml


def prepare(data_root, cache, stationary_arms=False, head_only=False):
    repo = Path(__file__).resolve().parents[2]
    raw = yaml.safe_load((repo / "unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml").read_text())
    config = {k: copy.deepcopy(v) for k, v in raw.items() if not k.startswith(".") and k != "datasets"}
    source = copy.deepcopy(raw[".unitree_fullbody_base"])
    source.update(name="UnifoLM_WBT", data_path="UnifoLM_WBT_Dataset",
                  precollected_stats_path=str(repo / "unifolm_wla/dataloader/multi_source_dataset/stats"))
    source["image_keys"] = copy.deepcopy(raw[".unitree_img_wo_stereo"])
    if head_only:
        source["image_keys"] = source["image_keys"][:1]
    if stationary_arms:
        source["action_keys"] = {k: v for k, v in source["action_keys"].items()
                                 if k in ("left_ee_pose", "right_ee_pose", "left_fig6d", "right_fig6d")}
    config.update(data_base=str(Path(data_root).resolve()), cache_dir=str(Path(cache).resolve()), datasets=[source])
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stationary-arms", action="store_true")
    parser.add_argument("--head-only", action="store_true")
    args = parser.parse_args()
    config = prepare(args.data_root, args.cache, args.stationary_arms, args.head_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.out)


if __name__ == "__main__":
    main()
