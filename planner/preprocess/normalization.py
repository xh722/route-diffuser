"""Simple normalization helpers for planner tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MinMaxNormalizer:
    """Normalize features from a known range into [-1, 1]."""

    minimum: torch.Tensor
    maximum: torch.Tensor

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        minimum = torch.as_tensor(self.minimum, device=values.device, dtype=values.dtype)
        maximum = torch.as_tensor(self.maximum, device=values.device, dtype=values.dtype)
        scaled = (values - minimum) / torch.clamp(maximum - minimum, min=1e-6)
        return scaled * 2.0 - 1.0

    def unnormalize(self, values: torch.Tensor) -> torch.Tensor:
        minimum = torch.as_tensor(self.minimum, device=values.device, dtype=values.dtype)
        maximum = torch.as_tensor(self.maximum, device=values.device, dtype=values.dtype)
        scaled = (values + 1.0) / 2.0
        return scaled * (maximum - minimum) + minimum
