"""Synthetic planning dataset built on top of the adapter boundary."""

from __future__ import annotations

import torch

from planner.datasets.adapters import (
    AdapterBackedPlanningDataset,
    DatasetManifest,
    SceneAdapter,
    SyntheticDatasetConfig,
    SyntheticSceneAdapter,
    build_synthetic_manifest,
)
from planner.datasets.schema import CanonicalSceneBatch


class SyntheticPlanningDataset(AdapterBackedPlanningDataset):
    """Backwards-compatible synthetic dataset wrapper for the planning pipeline."""

    def __init__(
        self,
        config: SyntheticDatasetConfig,
        manifest: DatasetManifest | None = None,
    ) -> None:
        self.config = config
        self.manifest = manifest
        super().__init__(SyntheticSceneAdapter(config=config, manifest=manifest))


class AdapterPlanningDataset(AdapterBackedPlanningDataset):
    """Generic planning dataset wrapper for adapter-backed public formats."""

    def __init__(
        self,
        *,
        config: object,
        adapter: SceneAdapter,
        manifest: DatasetManifest | None = None,
    ) -> None:
        self.config = config
        self.manifest = manifest
        super().__init__(adapter)


def collate_scene_batches(
    samples: list[dict[str, torch.Tensor | int | str | float | bool]]
) -> CanonicalSceneBatch:
    """Collate adapter samples into the canonical batched scene schema."""

    batch = CanonicalSceneBatch(
        ego_current_state=torch.stack([sample["ego_current_state"] for sample in samples]),
        neighbor_history=torch.stack([sample["neighbor_history"] for sample in samples]),
        neighbor_history_mask=torch.stack(
            [sample["neighbor_history_mask"] for sample in samples]
        ),
        lane_polylines=torch.stack([sample["lane_polylines"] for sample in samples]),
        lane_polylines_mask=torch.stack(
            [sample["lane_polylines_mask"] for sample in samples]
        ),
        route_lanes=torch.stack([sample["route_lanes"] for sample in samples]),
        route_lanes_mask=torch.stack([sample["route_lanes_mask"] for sample in samples]),
        future_ego_trajectory=torch.stack(
            [sample["future_ego_trajectory"] for sample in samples]
        ),
        future_ego_mask=torch.stack([sample["future_ego_mask"] for sample in samples]),
        metadata={
            "indices": [int(sample["index"]) for sample in samples],
            "scenario_names": [str(sample["scenario_name"]) for sample in samples],
            "entry_ids": [str(sample.get("entry_id", sample["index"])) for sample in samples],
        },
    )
    return batch.validate()


__all__ = [
    "AdapterPlanningDataset",
    "SyntheticDatasetConfig",
    "SyntheticPlanningDataset",
    "build_synthetic_manifest",
    "collate_scene_batches",
]
