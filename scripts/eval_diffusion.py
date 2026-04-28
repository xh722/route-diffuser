#!/usr/bin/env python3
"""Evaluate the diffusion planner on the synthetic smoke dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from planner.common import load_yaml_config, set_seed
from planner.datasets import (
    SyntheticDatasetConfig,
    SyntheticPlanningDataset,
    collate_scene_batches,
)
from planner.models import DiffusionPlanner, DiffusionPlannerConfig
from planner.trainers import evaluate_model_detailed


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

    output_dir = Path(infer_config.get("eval_output_dir", "outputs/eval"))
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

    model = DiffusionPlanner(DiffusionPlannerConfig.from_mapping(model_config))
    device = resolve_device(str(infer_config.get("device", "auto")))
    model = model.to(device)

    checkpoint_path = args.checkpoint or str(infer_config.get("checkpoint_path", ""))
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_model_detailed(
        model=model,
        dataloader=dataloader,
        device=device,
        num_samples=int(infer_config.get("num_samples", 1)),
        time_delta=dataset_config.time_delta,
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
