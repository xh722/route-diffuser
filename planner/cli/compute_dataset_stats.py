"""Public dataset statistics entry point for RouteDiffuser."""

from __future__ import annotations

import argparse
from pathlib import Path

from planner.cli.common import apply_overrides
from planner.common import load_yaml_config
from planner.datasets import build_dataset_from_config, compute_dataset_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=None)
    return parser.parse_args()


def default_stats_path(data_config: dict[str, object]) -> Path:
    dataset_type = str(data_config.get("dataset_type", "dataset"))
    dataset_name = str(data_config.get("dataset_name", dataset_type))
    return Path("outputs/stats") / f"{dataset_name}_stats.json"


def main() -> None:
    args = parse_args()
    data_config = load_yaml_config(args.data_config)
    data_config = apply_overrides(data_config, manifest_path=args.manifest_path)
    dataset = build_dataset_from_config(data_config)

    stats = compute_dataset_statistics(
        dataset,
        dataset_name=str(getattr(dataset.config, "dataset_name", "unknown_dataset")),
        dataset_type=str(data_config.get("dataset_type", "unknown")),
        batch_size=args.batch_size,
        max_scenes=args.max_scenes,
    )
    output_path = Path(args.output) if args.output else default_stats_path(data_config)
    stats.save(output_path)

    print(f"saved dataset statistics to {output_path}")
    print(f"scene_count={stats.scene_count} tensor_groups={len(stats.tensors)}")


if __name__ == "__main__":
    main()
