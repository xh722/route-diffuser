"""Diffusion training losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def noise_prediction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-squared error between predicted and target diffusion noise."""

    if mask is None:
        return F.mse_loss(predicted, target)
    if mask.shape != predicted.shape[:-1]:
        raise ValueError("mask shape must match predicted batch and horizon dimensions")

    loss = F.mse_loss(predicted, target, reduction="none")
    weights = mask.to(loss.dtype).unsqueeze(-1)
    return (loss * weights).sum() / weights.sum().clamp(min=1.0) / predicted.shape[-1]
