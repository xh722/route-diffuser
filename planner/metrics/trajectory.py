"""Open-loop metrics for trajectory planning."""

from __future__ import annotations

import torch


def _xy_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(predicted[..., :2] - target[..., :2], dim=-1)


def average_displacement_error(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean displacement error over the full trajectory horizon."""

    errors = _xy_error(predicted, target)
    if mask is None:
        return errors.mean()
    weights = mask.to(errors.dtype)
    return (errors * weights).sum() / weights.sum().clamp(min=1.0)


def final_displacement_error(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Final-step displacement error."""

    errors = _xy_error(predicted, target)
    if mask is None:
        return errors[:, -1].mean()

    last_valid = mask.long().sum(dim=1).clamp(min=1) - 1
    batch_index = torch.arange(errors.shape[0], device=errors.device)
    return errors[batch_index, last_valid].mean()


def route_consistency_error(
    predicted: torch.Tensor,
    route_polylines: torch.Tensor,
    route_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean minimum XY distance from predicted points to valid route polyline points."""

    predicted_xy = predicted[..., :2]
    route_xy = route_polylines[..., :2].reshape(route_polylines.shape[0], -1, 2)
    valid_route = route_mask.reshape(route_mask.shape[0], -1).bool()

    distances = torch.cdist(predicted_xy, route_xy)
    distances = distances.masked_fill(~valid_route.unsqueeze(1), float("inf"))
    min_distance = distances.min(dim=-1).values
    finite_mask = torch.isfinite(min_distance)
    if not finite_mask.any():
        return predicted.new_tensor(float("nan"))
    return min_distance[finite_mask].mean()


def summarize_open_loop_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    route_polylines: torch.Tensor,
    route_mask: torch.Tensor,
    future_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize key open-loop trajectory metrics."""

    ade = average_displacement_error(predicted, target, future_mask)
    fde = final_displacement_error(predicted, target, future_mask)
    route_error = route_consistency_error(predicted, route_polylines, route_mask)
    return {
        "ade": float(ade.item()),
        "fde": float(fde.item()),
        "route_error": float(route_error.item()),
    }
