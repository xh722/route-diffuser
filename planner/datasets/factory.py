"""Factory helpers for adapter-backed datasets and manifests."""

from __future__ import annotations

from typing import Any

from planner.datasets.adapters import DatasetManifest
from planner.datasets.synthetic import (
    SyntheticDatasetConfig,
    SyntheticPlanningDataset,
    build_synthetic_manifest,
)


def build_dataset_from_config(values: dict[str, Any]) -> SyntheticPlanningDataset:
    """Build a dataset from a config mapping."""

    manifest = _load_manifest_from_config(values)
    dataset_type = _resolve_dataset_type(values, manifest)

    if dataset_type == "synthetic":
        config_values = {}
        if manifest is not None:
            config_values.update(manifest.config)
        config_values.update(values)
        config = SyntheticDatasetConfig.from_mapping(config_values)
        return SyntheticPlanningDataset(config=config, manifest=manifest)

    raise ValueError(f"Unsupported dataset_type: {dataset_type!r}")


def build_manifest_from_config(
    values: dict[str, Any],
    split: str = "train",
    start_index: int = 0,
    num_samples: int | None = None,
) -> DatasetManifest:
    """Build a manifest from a config mapping."""

    dataset_type = str(values.get("dataset_type", "synthetic")).strip() or "synthetic"
    if dataset_type == "synthetic":
        config = SyntheticDatasetConfig.from_mapping(values)
        return build_synthetic_manifest(
            config=config,
            split=split,
            start_index=start_index,
            num_samples=num_samples,
        )

    raise ValueError(f"Unsupported dataset_type: {dataset_type!r}")


def _load_manifest_from_config(values: dict[str, Any]) -> DatasetManifest | None:
    manifest_path = str(values.get("manifest_path", "")).strip()
    if not manifest_path:
        return None
    return DatasetManifest.load(manifest_path)


def _resolve_dataset_type(
    values: dict[str, Any],
    manifest: DatasetManifest | None,
) -> str:
    dataset_type = str(values.get("dataset_type", "")).strip()
    if dataset_type:
        return dataset_type
    if manifest is not None:
        return manifest.adapter_type
    return "synthetic"
