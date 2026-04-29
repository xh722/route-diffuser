"""Dataset statistics and caching helpers for canonical planning tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from planner.datasets.schema import CanonicalSceneBatch
from planner.datasets.synthetic import collate_scene_batches


@dataclass
class TensorStatistics:
    """Per-feature summary statistics for one tensor family."""

    count: int
    mean: list[float]
    std: list[float]
    minimum: list[float]
    maximum: list[float]

    @classmethod
    def from_tensors(cls, values: torch.Tensor) -> "TensorStatistics":
        if values.ndim != 2:
            raise ValueError(f"Expected rank-2 values for statistics, got {values.ndim}")
        if values.shape[0] == 0:
            raise ValueError("Cannot compute statistics from zero rows")

        values = values.to(torch.float64)
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False)
        minimum = values.min(dim=0).values
        maximum = values.max(dim=0).values
        return cls(
            count=int(values.shape[0]),
            mean=mean.tolist(),
            std=std.tolist(),
            minimum=minimum.tolist(),
            maximum=maximum.tolist(),
        )

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TensorStatistics":
        return cls(
            count=int(values["count"]),
            mean=[float(item) for item in values["mean"]],
            std=[float(item) for item in values["std"]],
            minimum=[float(item) for item in values["minimum"]],
            maximum=[float(item) for item in values["maximum"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass
class DatasetStatistics:
    """Serializable cached statistics for one dataset configuration or manifest."""

    dataset_name: str
    dataset_type: str
    scene_count: int
    tensors: dict[str, TensorStatistics] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "DatasetStatistics":
        return cls(
            dataset_name=str(values["dataset_name"]),
            dataset_type=str(values["dataset_type"]),
            scene_count=int(values["scene_count"]),
            tensors={
                str(key): TensorStatistics.from_mapping(stat_values)
                for key, stat_values in dict(values.get("tensors", {})).items()
            },
            metadata=dict(values.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DatasetStatistics":
        stats_path = Path(path)
        with stats_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected mapping at {stats_path}")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type,
            "scene_count": self.scene_count,
            "tensors": {
                key: value.to_dict() for key, value in sorted(self.tensors.items())
            },
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        stats_path = Path(path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return stats_path


def _masked_feature_rows(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.reshape(-1, values.shape[-1])
    rows = values[mask.bool()]
    if rows.ndim == 1:
        rows = rows.unsqueeze(0)
    return rows


def _collect_tensor_rows(batch: CanonicalSceneBatch) -> dict[str, torch.Tensor]:
    rows: dict[str, torch.Tensor] = {
        "ego_current_state": _masked_feature_rows(batch.ego_current_state, None),
        "neighbor_history": _masked_feature_rows(
            batch.neighbor_history, batch.neighbor_history_mask
        ),
        "lane_polylines": _masked_feature_rows(
            batch.lane_polylines, batch.lane_polylines_mask
        ),
        "route_lanes": _masked_feature_rows(batch.route_lanes, batch.route_lanes_mask),
    }
    if batch.future_ego_trajectory is not None:
        rows["future_ego_trajectory"] = _masked_feature_rows(
            batch.future_ego_trajectory,
            batch.future_ego_mask,
        )
    return rows


def compute_dataset_statistics(
    dataset,
    *,
    dataset_name: str,
    dataset_type: str,
    batch_size: int = 8,
    max_scenes: int | None = None,
) -> DatasetStatistics:
    """Compute reusable per-feature statistics for a dataset."""

    scene_count = len(dataset) if max_scenes is None else min(len(dataset), max_scenes)
    if scene_count <= 0:
        raise ValueError("Dataset statistics require at least one scene")

    samples = [dataset[index] for index in range(scene_count)]
    dataloader = DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_scene_batches,
    )

    collected: dict[str, list[torch.Tensor]] = {}
    for batch in dataloader:
        batch_rows = _collect_tensor_rows(batch)
        for key, rows in batch_rows.items():
            if rows.numel() == 0:
                continue
            collected.setdefault(key, []).append(rows.detach().cpu())

    tensor_stats = {
        key: TensorStatistics.from_tensors(torch.cat(rows, dim=0))
        for key, rows in collected.items()
    }
    return DatasetStatistics(
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        scene_count=scene_count,
        tensors=tensor_stats,
        metadata={
            "batch_size": batch_size,
            "max_scenes": scene_count,
        },
    )
