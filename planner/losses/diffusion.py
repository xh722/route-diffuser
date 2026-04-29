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


def candidate_score_distillation_loss(
    predicted_scores: torch.Tensor,
    target_costs: torch.Tensor,
    *,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Match predicted candidate preferences to oracle cost-based preferences."""

    if predicted_scores.shape != target_costs.shape:
        raise ValueError("predicted_scores and target_costs must share the same shape")
    if predicted_scores.ndim != 2:
        raise ValueError("candidate score distillation expects shape [B, S]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    target_logits = -target_costs / temperature
    target_probs = torch.softmax(target_logits, dim=1)
    predicted_log_probs = torch.log_softmax(predicted_scores, dim=1)
    return -(target_probs * predicted_log_probs).sum(dim=1).mean()
