"""Open-loop metrics for trajectory planning."""

from __future__ import annotations

import torch

from planner.metrics.collision import box_collision_matrix


def _xy_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(predicted[..., :2] - target[..., :2], dim=-1)


def _last_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    return mask.long().sum(dim=1).clamp(min=1) - 1


def _gather_last_valid(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    indices = _last_valid_indices(mask)
    batch_index = torch.arange(values.shape[0], device=values.device)
    return values[batch_index, indices]


def average_displacement_errors(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-trajectory mean displacement error."""

    errors = _xy_error(predicted, target)
    if mask is None:
        return errors.mean(dim=1)
    weights = mask.to(errors.dtype)
    return (errors * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def average_displacement_error(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean displacement error over the full trajectory horizon."""

    return average_displacement_errors(predicted, target, mask).mean()


def final_displacement_errors(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-trajectory final-step displacement error."""

    errors = _xy_error(predicted, target)
    if mask is None:
        return errors[:, -1]

    last_valid = _last_valid_indices(mask)
    batch_index = torch.arange(errors.shape[0], device=errors.device)
    return errors[batch_index, last_valid]


def final_displacement_error(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Final-step displacement error."""

    return final_displacement_errors(predicted, target, mask).mean()


def route_consistency_errors(
    predicted: torch.Tensor,
    route_polylines: torch.Tensor,
    route_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-trajectory route consistency error."""

    predicted_xy = predicted[..., :2]
    route_xy = route_polylines[..., :2].reshape(route_polylines.shape[0], -1, 2)
    valid_route = route_mask.reshape(route_mask.shape[0], -1).bool()

    distances = torch.cdist(predicted_xy, route_xy)
    distances = distances.masked_fill(~valid_route.unsqueeze(1), float("inf"))
    min_distance = distances.min(dim=-1).values
    finite_mask = torch.isfinite(min_distance)
    masked_distances = torch.where(finite_mask, min_distance, torch.zeros_like(min_distance))
    counts = finite_mask.sum(dim=1).clamp(min=1)
    per_trajectory = masked_distances.sum(dim=1) / counts
    nan_fill = torch.full_like(per_trajectory, float("nan"))
    return torch.where(finite_mask.any(dim=1), per_trajectory, nan_fill)


def route_consistency_error(
    predicted: torch.Tensor,
    route_polylines: torch.Tensor,
    route_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean minimum XY distance from predicted points to valid route polyline points."""

    per_trajectory = route_consistency_errors(predicted, route_polylines, route_mask)
    finite_mask = torch.isfinite(per_trajectory)
    if not finite_mask.any():
        return predicted.new_tensor(float("nan"))
    return per_trajectory[finite_mask].mean()


def trajectory_progress(
    predicted: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-trajectory forward progress using the x coordinate."""

    if mask is None:
        final_x = predicted[:, -1, 0]
    else:
        final_x = _gather_last_valid(predicted[..., 0], mask)
    return final_x - predicted[:, 0, 0]


def trajectory_comfort_metrics(
    predicted: torch.Tensor,
    time_delta: float,
) -> dict[str, torch.Tensor]:
    """Per-trajectory comfort-oriented kinematic metrics."""

    if time_delta <= 0.0:
        raise ValueError("time_delta must be positive")

    if predicted.shape[1] > 1:
        velocity_xy = torch.diff(predicted[..., :2], dim=1) / time_delta
        speed = torch.linalg.norm(velocity_xy, dim=-1)
    else:
        velocity_xy = predicted.new_zeros(predicted.shape[0], 0, 2)
        speed = predicted.new_zeros(predicted.shape[0], 0)

    if speed.shape[1] > 1:
        longitudinal_accel = torch.diff(speed, dim=1) / time_delta
        heading = torch.atan2(velocity_xy[..., 1], velocity_xy[..., 0])
        heading_delta = torch.atan2(
            torch.sin(torch.diff(heading, dim=1)),
            torch.cos(torch.diff(heading, dim=1)),
        )
        yaw_rate = heading_delta / time_delta
        lateral_accel = speed[:, 1:] * yaw_rate
    else:
        longitudinal_accel = predicted.new_zeros(predicted.shape[0], 0)
        yaw_rate = predicted.new_zeros(predicted.shape[0], 0)
        lateral_accel = predicted.new_zeros(predicted.shape[0], 0)

    if longitudinal_accel.shape[1] > 1:
        jerk = torch.diff(longitudinal_accel, dim=1) / time_delta
    else:
        jerk = predicted.new_zeros(predicted.shape[0], 0)

    zeros = predicted.new_zeros(predicted.shape[0])
    max_forward_accel = (
        longitudinal_accel.max(dim=1).values if longitudinal_accel.numel() else zeros
    )
    max_brake = (
        longitudinal_accel.min(dim=1).values if longitudinal_accel.numel() else zeros
    )
    max_abs_lateral_accel = (
        lateral_accel.abs().max(dim=1).values if lateral_accel.numel() else zeros
    )
    max_abs_jerk = jerk.abs().max(dim=1).values if jerk.numel() else zeros
    max_abs_yaw_rate = yaw_rate.abs().max(dim=1).values if yaw_rate.numel() else zeros
    comfort_violation = (
        (max_forward_accel > 2.40)
        | (max_brake < -4.05)
        | (max_abs_lateral_accel > 4.89)
        | (max_abs_jerk > 4.13)
        | (max_abs_yaw_rate > 0.95)
    )

    return {
        "max_forward_accel": max_forward_accel,
        "max_brake": max_brake,
        "max_abs_lateral_accel": max_abs_lateral_accel,
        "max_abs_jerk": max_abs_jerk,
        "max_abs_yaw_rate": max_abs_yaw_rate,
        "comfort_violation": comfort_violation,
    }


def neighbor_clearance_metrics(
    predicted: torch.Tensor,
    neighbor_history: torch.Tensor,
    neighbor_history_mask: torch.Tensor,
    time_delta: float,
    collision_distance: float = 2.0,
    ego_length: float = 4.8,
    ego_width: float = 2.0,
    neighbor_length: float = 4.8,
    neighbor_width: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Per-trajectory clearance metrics using linear neighbor extrapolation."""

    if time_delta <= 0.0:
        raise ValueError("time_delta must be positive")

    valid_neighbors = neighbor_history_mask.any(dim=-1)
    last_indices = neighbor_history_mask.long().sum(dim=-1).clamp(min=1) - 1
    gather_index = last_indices.unsqueeze(-1).unsqueeze(-1).expand(
        -1, -1, 1, neighbor_history.shape[-1]
    )
    last_state = torch.gather(neighbor_history, dim=2, index=gather_index).squeeze(2)

    horizon = predicted.shape[1]
    time_offsets = (
        torch.arange(horizon, device=predicted.device, dtype=predicted.dtype) * time_delta
    )
    future_neighbor_xy = (
        last_state[..., :2].unsqueeze(2)
        + last_state[..., 4:6].unsqueeze(2) * time_offsets.view(1, 1, -1, 1)
    )
    future_neighbor_states = last_state.unsqueeze(2).expand(-1, -1, horizon, -1).clone()
    future_neighbor_states[..., :2] = future_neighbor_xy

    ego_xy = predicted[..., :2].unsqueeze(1)
    distances = torch.linalg.norm(ego_xy - future_neighbor_xy, dim=-1)
    distances = distances.masked_fill(~valid_neighbors.unsqueeze(-1), float("inf"))

    min_clearance = distances.amin(dim=(1, 2))
    min_clearance = torch.where(
        torch.isfinite(min_clearance),
        min_clearance,
        torch.full_like(min_clearance, float("inf")),
    )
    point_collision = (distances < collision_distance).any(dim=(1, 2))
    box_collision = box_collision_matrix(
        predicted,
        future_neighbor_states,
        ego_length=ego_length,
        ego_width=ego_width,
        object_length=neighbor_length,
        object_width=neighbor_width,
    )
    box_collision = box_collision & valid_neighbors.unsqueeze(-1)
    collision = box_collision.any(dim=(1, 2))
    return {
        "min_clearance": min_clearance,
        "point_collision": point_collision,
        "box_collision": collision,
        "collision": collision,
    }


def compute_open_loop_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    route_polylines: torch.Tensor,
    route_mask: torch.Tensor,
    future_mask: torch.Tensor | None = None,
    neighbor_history: torch.Tensor | None = None,
    neighbor_history_mask: torch.Tensor | None = None,
    time_delta: float = 1.0 / 3.0,
) -> dict[str, torch.Tensor]:
    """Compute per-trajectory planning metrics."""

    metrics: dict[str, torch.Tensor] = {
        "ade": average_displacement_errors(predicted, target, future_mask),
        "fde": final_displacement_errors(predicted, target, future_mask),
        "route_error": route_consistency_errors(predicted, route_polylines, route_mask),
        "progress": trajectory_progress(predicted, future_mask),
    }
    metrics.update(trajectory_comfort_metrics(predicted, time_delta=time_delta))
    if neighbor_history is not None and neighbor_history_mask is not None:
        metrics.update(
            neighbor_clearance_metrics(
                predicted,
                neighbor_history=neighbor_history,
                neighbor_history_mask=neighbor_history_mask,
                time_delta=time_delta,
            )
        )
    return metrics


def candidate_set_metrics(
    predicted_samples: torch.Tensor,
    target: torch.Tensor,
    route_polylines: torch.Tensor,
    route_mask: torch.Tensor,
    future_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-scene candidate-set metrics."""

    if predicted_samples.ndim != 4:
        raise ValueError("predicted_samples must have shape [B, S, T, D]")

    batch_size, num_samples, horizon, state_dim = predicted_samples.shape
    repeated_target = target.unsqueeze(1).expand(-1, num_samples, -1, -1).reshape(
        batch_size * num_samples, horizon, state_dim
    )
    repeated_mask = None
    if future_mask is not None:
        repeated_mask = future_mask.unsqueeze(1).expand(-1, num_samples, -1).reshape(
            batch_size * num_samples, horizon
        )
    repeated_routes = route_polylines.unsqueeze(1).expand(
        -1, num_samples, -1, -1, -1
    ).reshape(batch_size * num_samples, route_polylines.shape[1], route_polylines.shape[2], route_polylines.shape[3])
    repeated_route_mask = route_mask.unsqueeze(1).expand(
        -1, num_samples, -1, -1
    ).reshape(batch_size * num_samples, route_mask.shape[1], route_mask.shape[2])
    flattened_predictions = predicted_samples.reshape(batch_size * num_samples, horizon, state_dim)

    ade = average_displacement_errors(flattened_predictions, repeated_target, repeated_mask).reshape(
        batch_size, num_samples
    )
    fde = final_displacement_errors(flattened_predictions, repeated_target, repeated_mask).reshape(
        batch_size, num_samples
    )
    route_error = route_consistency_errors(
        flattened_predictions, repeated_routes, repeated_route_mask
    ).reshape(batch_size, num_samples)

    final_xy = predicted_samples[..., -1, :2]
    if num_samples > 1:
        pairwise_distances = torch.cdist(final_xy, final_xy)
        upper_mask = torch.triu(
            torch.ones(num_samples, num_samples, device=predicted_samples.device, dtype=torch.bool),
            diagonal=1,
        )
        diversity = pairwise_distances[:, upper_mask].mean(dim=1)
    else:
        diversity = predicted_samples.new_zeros(batch_size)

    return {
        "oracle_ade": ade.min(dim=1).values,
        "oracle_fde": fde.min(dim=1).values,
        "oracle_route_error": route_error.min(dim=1).values,
        "candidate_final_diversity": diversity,
    }


def _mean_metric(metric: torch.Tensor) -> float:
    finite_mask = torch.isfinite(metric)
    if not finite_mask.any():
        return float("nan")
    return float(metric[finite_mask].mean().item())


def summarize_open_loop_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    route_polylines: torch.Tensor,
    route_mask: torch.Tensor,
    future_mask: torch.Tensor | None = None,
    neighbor_history: torch.Tensor | None = None,
    neighbor_history_mask: torch.Tensor | None = None,
    time_delta: float = 1.0 / 3.0,
) -> dict[str, float]:
    """Summarize key open-loop trajectory metrics."""

    metrics = compute_open_loop_metrics(
        predicted=predicted,
        target=target,
        route_polylines=route_polylines,
        route_mask=route_mask,
        future_mask=future_mask,
        neighbor_history=neighbor_history,
        neighbor_history_mask=neighbor_history_mask,
        time_delta=time_delta,
    )
    summary = {
        "ade": _mean_metric(metrics["ade"]),
        "fde": _mean_metric(metrics["fde"]),
        "route_error": _mean_metric(metrics["route_error"]),
        "progress": _mean_metric(metrics["progress"]),
        "max_forward_accel": _mean_metric(metrics["max_forward_accel"]),
        "max_brake": _mean_metric(metrics["max_brake"]),
        "max_abs_lateral_accel": _mean_metric(metrics["max_abs_lateral_accel"]),
        "max_abs_jerk": _mean_metric(metrics["max_abs_jerk"]),
        "max_abs_yaw_rate": _mean_metric(metrics["max_abs_yaw_rate"]),
        "comfort_violation_rate": float(metrics["comfort_violation"].to(torch.float32).mean().item()),
    }
    if "min_clearance" in metrics:
        summary["min_clearance"] = _mean_metric(metrics["min_clearance"])
        summary["collision_rate"] = float(metrics["collision"].to(torch.float32).mean().item())
        summary["box_collision_rate"] = float(
            metrics["box_collision"].to(torch.float32).mean().item()
        )
        summary["point_collision_rate"] = float(
            metrics["point_collision"].to(torch.float32).mean().item()
        )
    return summary
