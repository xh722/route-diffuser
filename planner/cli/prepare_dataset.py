"""Public dataset preparation entry point for RouteDiffuser."""

from __future__ import annotations

import argparse
from pathlib import Path

from planner.common import load_yaml_config
from planner.datasets import build_manifest_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None)
    return parser.parse_args()


def default_manifest_path(data_config: dict[str, object], split: str) -> Path:
    dataset_type = str(data_config.get("dataset_type", "synthetic"))
    dataset_name = str(data_config.get("dataset_name", dataset_type))
    return Path("outputs/manifests") / f"{dataset_name}_{split}.json"


def main() -> None:
    args = parse_args()
    data_config = load_yaml_config(args.data_config)
    manifest = build_manifest_from_config(
        data_config,
        split=args.split,
        start_index=args.start_index,
        num_samples=args.num_samples,
    )

    output_path = (
        Path(args.output)
        if args.output
        else default_manifest_path(data_config, args.split)
    )
    manifest.save(output_path)
    print(f"saved manifest to {output_path}")
    print(
        f"adapter_type={manifest.adapter_type} split={manifest.split} scene_count={manifest.scene_count}"
    )


if __name__ == "__main__":
    main()
