"""Export canonical planning samples into the public NPZ bridge format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from planner.cli.common import load_dataset_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, dataset, _, _ = load_dataset_bundle(
        args.data_config,
        batch_size=1,
        shuffle=False,
        manifest_path=args.manifest_path,
    )

    samples = [dataset[index] for index in range(len(dataset))]
    payload = {
        "ego_current_state": np.stack(
            [np.asarray(sample["ego_current_state"], dtype=np.float32) for sample in samples]
        ),
        "neighbor_history": np.stack(
            [np.asarray(sample["neighbor_history"], dtype=np.float32) for sample in samples]
        ),
        "neighbor_history_mask": np.stack(
            [np.asarray(sample["neighbor_history_mask"], dtype=bool) for sample in samples]
        ),
        "lane_polylines": np.stack(
            [np.asarray(sample["lane_polylines"], dtype=np.float32) for sample in samples]
        ),
        "lane_polylines_mask": np.stack(
            [np.asarray(sample["lane_polylines_mask"], dtype=bool) for sample in samples]
        ),
        "route_lanes": np.stack(
            [np.asarray(sample["route_lanes"], dtype=np.float32) for sample in samples]
        ),
        "route_lanes_mask": np.stack(
            [np.asarray(sample["route_lanes_mask"], dtype=bool) for sample in samples]
        ),
        "future_ego_trajectory": np.stack(
            [np.asarray(sample["future_ego_trajectory"], dtype=np.float32) for sample in samples]
        ),
        "future_ego_mask": np.stack(
            [np.asarray(sample["future_ego_mask"], dtype=bool) for sample in samples]
        ),
        "scenario_name": np.asarray(
            [str(sample["scenario_name"]) for sample in samples],
            dtype=object,
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    print(f"saved npz dataset to {output_path}")
    print(f"scene_count={len(samples)}")


if __name__ == "__main__":
    main()
