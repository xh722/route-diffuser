#!/usr/bin/env python3
"""Generate synthetic planner trajectories for smoke inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from planner.common import load_yaml_config, set_seed
from planner.datasets import (
    SyntheticDatasetConfig,
    SyntheticPlanningDataset,
    collate_scene_batches,
)
from planner.inference import score_trajectory_candidates
from planner.models import DiffusionPlanner, DiffusionPlannerConfig
from planner.visualization import plot_trajectory_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/base.yaml")
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--inference-config", default="configs/inference/base.yaml")
    parser.add_argument("--checkpoint", default="")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main() -> None:
    args = parse_args()
    model_config = load_yaml_config(args.model_config)
    data_config = load_yaml_config(args.data_config)
    infer_config = load_yaml_config(args.inference_config)

    output_dir = Path(infer_config.get("output_dir", "outputs/infer"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(infer_config.get("seed", 7)))

    dataset_config = SyntheticDatasetConfig.from_mapping(data_config)
    dataset = SyntheticPlanningDataset(dataset_config)
    dataloader = DataLoader(
        dataset,
        batch_size=int(infer_config.get("batch_size", 4)),
        shuffle=False,
        collate_fn=collate_scene_batches,
    )
    batch = next(iter(dataloader))

    model = DiffusionPlanner(DiffusionPlannerConfig.from_mapping(model_config))
    device = resolve_device(str(infer_config.get("device", "auto")))
    model = model.to(device)

    checkpoint_path = args.checkpoint or str(infer_config.get("checkpoint_path", ""))
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    batch = batch.to(device)
    num_samples = int(infer_config.get("num_samples", 3))
    predictions = model.sample(batch, num_samples=num_samples)
    scored = score_trajectory_candidates(
        predicted_samples=predictions,
        scene_batch=batch,
        time_delta=dataset_config.time_delta,
    )

    prediction_path = output_dir / "predictions.pt"
    torch.save(predictions.cpu(), prediction_path)
    plot_trajectory_comparison(
        predicted=scored["selected_trajectories"],
        target=batch.future_ego_trajectory,
        route_polylines=batch.route_lanes,
        route_mask=batch.route_lanes_mask,
        output_path=output_dir / "prediction_plot.png",
    )
    print(f"saved predictions to {prediction_path}")


if __name__ == "__main__":
    main()
