"""Base interfaces for pluggable dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import torch
from torch.utils.data import Dataset

SceneSample: TypeAlias = dict[str, torch.Tensor | int | str | float | bool]


class SceneAdapter(ABC):
    """Adapter interface for scene-oriented planning datasets."""

    @property
    @abstractmethod
    def adapter_type(self) -> str:
        """Return the adapter type string used in manifests and configs."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of addressable scene samples."""

    @abstractmethod
    def get_sample(self, index: int) -> SceneSample:
        """Return one canonical scene sample."""


class AdapterBackedPlanningDataset(Dataset[SceneSample]):
    """Thin dataset wrapper that delegates storage and decoding to an adapter."""

    def __init__(self, adapter: SceneAdapter) -> None:
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.adapter)

    def __getitem__(self, index: int) -> SceneSample:
        return self.adapter.get_sample(index)
