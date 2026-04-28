"""Diffusion training losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def noise_prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared error between predicted and target diffusion noise."""

    return F.mse_loss(predicted, target)
