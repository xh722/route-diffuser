"""Shared helpers for public planner CLI entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from planner.common import load_yaml_config
from planner.datasets import build_dataset_from_config, collate_scene_batches
from planner.models import DiffusionPlanner, DiffusionPlannerConfig


def resolve_device(device_name: str) -> torch.device:
    """Resolve a user-facing device name into a torch device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def apply_overrides(
    config: dict[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
    """Return a config mapping with non-empty runtime overrides applied."""

    merged = dict(config)
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        merged[key] = value
    return merged


def load_dataset_bundle(
    data_config_path: str | Path,
    *,
    batch_size: int,
    shuffle: bool,
    manifest_path: str = "",
) -> tuple[dict[str, Any], Any, Any, DataLoader]:
    """Load a dataset config, dataset instance, and dataloader."""

    data_config = load_yaml_config(data_config_path)
    data_config = apply_overrides(data_config, manifest_path=manifest_path)
    dataset = build_dataset_from_config(data_config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_scene_batches,
    )
    return data_config, dataset, dataset.config, dataloader


def load_planner_model(
    model_config_path: str | Path,
    *,
    device: torch.device,
    checkpoint_path: str = "",
) -> tuple[dict[str, Any], DiffusionPlanner]:
    """Load a planner model and optionally restore a checkpoint."""

    model_config = load_yaml_config(model_config_path)
    model = DiffusionPlanner(DiffusionPlannerConfig.from_mapping(model_config))
    model = model.to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model_config, model
