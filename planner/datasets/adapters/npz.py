"""NPZ-backed public dataset adapter for canonical planning tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from planner.datasets.adapters.base import SceneAdapter, SceneSample
from planner.datasets.adapters.manifest import DatasetManifest, SceneManifestEntry


@dataclass(frozen=True)
class NpzDatasetConfig:
    """Configuration for NPZ-backed planning datasets."""

    dataset_name: str = "route_diffuser_npz"
    source_path: str = ""
    state_dim: int = 6
    lane_dim: int = 4
    time_delta: float = 1.0 / 3.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "NpzDatasetConfig":
        filtered = {key: values[key] for key in cls.__dataclass_fields__ if key in values}
        return cls(**filtered)


def build_npz_manifest(
    config: NpzDatasetConfig,
    split: str = "train",
) -> DatasetManifest:
    """Build a manifest from an NPZ container."""

    if not config.source_path:
        raise ValueError("NPZ manifest generation requires source_path")

    source_path = Path(config.source_path)
    with np.load(source_path, allow_pickle=True) as payload:
        sample_count = int(payload["ego_current_state"].shape[0])
        scenario_names = (
            payload["scenario_name"]
            if "scenario_name" in payload
            else np.array(["unknown"] * sample_count)
        )

    entries = []
    for sample_index in range(sample_count):
        scenario_name = str(scenario_names[sample_index])
        entries.append(
            SceneManifestEntry(
                entry_id=f"npz-{split}-{sample_index:06d}",
                source_path=str(source_path),
                sample_index=sample_index,
                scenario_name=scenario_name,
                tags=(split, scenario_name),
                metadata={},
            )
        )

    return DatasetManifest(
        dataset_name=config.dataset_name,
        adapter_type="npz",
        split=split,
        scene_count=len(entries),
        config=asdict(config),
        entries=entries,
    )


class NpzSceneAdapter(SceneAdapter):
    """Adapter that reads canonical planning tensors from one NPZ file."""

    REQUIRED_KEYS = (
        "ego_current_state",
        "neighbor_history",
        "neighbor_history_mask",
        "lane_polylines",
        "lane_polylines_mask",
        "route_lanes",
        "route_lanes_mask",
        "future_ego_trajectory",
        "future_ego_mask",
    )

    def __init__(
        self,
        config: NpzDatasetConfig,
        manifest: DatasetManifest | None = None,
    ) -> None:
        if manifest is not None and manifest.adapter_type != "npz":
            raise ValueError(f"Expected npz manifest, got {manifest.adapter_type!r}")
        if not config.source_path:
            raise ValueError("NPZ adapter requires source_path")

        self.config = config
        self.manifest = manifest
        self._payload = self._load_payload(Path(config.source_path))
        self._entries = self._build_entries()

    @property
    def adapter_type(self) -> str:
        return "npz"

    def __len__(self) -> int:
        return len(self._entries)

    def get_sample(self, index: int) -> SceneSample:
        entry = self._entries[index]
        sample_index = int(entry["sample_index"])
        scenario_name = str(entry["scenario_name"])

        return {
            "ego_current_state": self._tensor("ego_current_state", sample_index),
            "neighbor_history": self._tensor("neighbor_history", sample_index),
            "neighbor_history_mask": self._tensor("neighbor_history_mask", sample_index, dtype=torch.bool),
            "lane_polylines": self._tensor("lane_polylines", sample_index),
            "lane_polylines_mask": self._tensor("lane_polylines_mask", sample_index, dtype=torch.bool),
            "route_lanes": self._tensor("route_lanes", sample_index),
            "route_lanes_mask": self._tensor("route_lanes_mask", sample_index, dtype=torch.bool),
            "future_ego_trajectory": self._tensor("future_ego_trajectory", sample_index),
            "future_ego_mask": self._tensor("future_ego_mask", sample_index, dtype=torch.bool),
            "index": sample_index,
            "scenario_name": scenario_name,
            "entry_id": str(entry["entry_id"]),
        }

    def _build_entries(self) -> list[dict[str, object]]:
        if self.manifest is not None:
            return [
                {
                    "entry_id": entry.entry_id,
                    "sample_index": 0 if entry.sample_index is None else int(entry.sample_index),
                    "scenario_name": entry.scenario_name or self._default_scenario_name(int(entry.sample_index or 0)),
                }
                for entry in self.manifest.entries
            ]

        sample_count = int(self._payload["ego_current_state"].shape[0])
        return [
            {
                "entry_id": f"npz-train-{sample_index:06d}",
                "sample_index": sample_index,
                "scenario_name": self._default_scenario_name(sample_index),
            }
            for sample_index in range(sample_count)
        ]

    def _default_scenario_name(self, sample_index: int) -> str:
        scenario_names = self._payload.get("scenario_name")
        if scenario_names is None:
            return "unknown"
        return str(scenario_names[sample_index])

    def _tensor(
        self,
        key: str,
        sample_index: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        value = self._payload[key][sample_index]
        tensor = torch.as_tensor(value)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    @classmethod
    def _load_payload(cls, path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=True) as payload:
            arrays = {key: payload[key] for key in payload.files}
        missing = [key for key in cls.REQUIRED_KEYS if key not in arrays]
        if missing:
            raise ValueError(f"NPZ dataset is missing required keys: {missing}")
        return arrays
