"""Heuristic candidate scoring inspired by multi-candidate planning stacks."""

from __future__ import annotations

import torch

from planner.datasets.schema import CanonicalSceneBatch
from planner.metrics.trajectory import (
    neighbor_clearance_metrics,
    route_consistency_errors,
    trajectory_comfort_metrics,
    trajectory_progress,
)

COMFORT_LIMITS = {
    "forward_accel": 2.40,
    "brake": 4.05,
    "lateral_accel": 4.89,
    "jerk": 4.13,
    "yaw_rate": 0.95,
}
CLEARANCE_MARGIN = 3.0
COLLISION_DISTANCE = 2.0


def _repeat_scene_tensor(tensor: torch.Tensor, num_samples: int) -> torch.Tensor:
    expand_shape = [tensor.shape[0], num_samples, *tensor.shape[1:]]
    return tensor.unsqueeze(1).expand(*expand_shape).reshape(-1, *tensor.shape[1:])


def _gather_candidates(
    predicted_samples: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    batch_index = torch.arange(predicted_samples.shape[0], device=predicted_samples.device)
    return predicted_samples[batch_index, indices]


@torch.no_grad()
def score_trajectory_candidates(
    predicted_samples: torch.Tensor,
    scene_batch: CanonicalSceneBatch,
    time_delta: float = 1.0 / 3.0,
) -> dict[str, torch.Tensor]:
    """Score each sampled trajectory using route, clearance, and comfort priors."""

    scene_batch.validate()
    if predicted_samples.ndim != 4:
        raise ValueError("predicted_samples must have shape [B, S, T, D]")

    batch_size, num_samples, horizon, state_dim = predicted_samples.shape
    flattened_predictions = predicted_samples.reshape(batch_size * num_samples, horizon, state_dim)

    route_error = route_consistency_errors(
        flattened_predictions,
        route_polylines=_repeat_scene_tensor(scene_batch.route_lanes, num_samples),
        route_mask=_repeat_scene_tensor(scene_batch.route_lanes_mask, num_samples),
    ).reshape(batch_size, num_samples)
    progress = trajectory_progress(flattened_predictions).reshape(batch_size, num_samples)
    comfort = trajectory_comfort_metrics(flattened_predictions, time_delta=time_delta)
    clearance = neighbor_clearance_metrics(
        flattened_predictions,
        neighbor_history=_repeat_scene_tensor(scene_batch.neighbor_history, num_samples),
        neighbor_history_mask=_repeat_scene_tensor(scene_batch.neighbor_history_mask, num_samples),
        time_delta=time_delta,
        collision_distance=COLLISION_DISTANCE,
    )

    max_forward_accel = comfort["max_forward_accel"].reshape(batch_size, num_samples)
    max_brake = comfort["max_brake"].reshape(batch_size, num_samples)
    max_abs_lateral_accel = comfort["max_abs_lateral_accel"].reshape(batch_size, num_samples)
    max_abs_jerk = comfort["max_abs_jerk"].reshape(batch_size, num_samples)
    max_abs_yaw_rate = comfort["max_abs_yaw_rate"].reshape(batch_size, num_samples)
    min_clearance = clearance["min_clearance"].reshape(batch_size, num_samples)
    collision = clearance["collision"].reshape(batch_size, num_samples)

    comfort_penalty = (
        torch.relu(max_forward_accel - COMFORT_LIMITS["forward_accel"]) / COMFORT_LIMITS["forward_accel"]
        + torch.relu(-COMFORT_LIMITS["brake"] - max_brake) / COMFORT_LIMITS["brake"]
        + torch.relu(max_abs_lateral_accel - COMFORT_LIMITS["lateral_accel"]) / COMFORT_LIMITS["lateral_accel"]
        + torch.relu(max_abs_jerk - COMFORT_LIMITS["jerk"]) / COMFORT_LIMITS["jerk"]
        + torch.relu(max_abs_yaw_rate - COMFORT_LIMITS["yaw_rate"]) / COMFORT_LIMITS["yaw_rate"]
    )
    clearance_penalty = (
        torch.relu(CLEARANCE_MARGIN - min_clearance) / CLEARANCE_MARGIN
        + collision.to(route_error.dtype) * 2.0
    )
    progress_reward = progress / progress.abs().amax(dim=1, keepdim=True).clamp(min=1.0)

    scores = (
        route_error
        + 0.75 * clearance_penalty
        + 0.35 * comfort_penalty
        - 0.10 * progress_reward
    )
    selected_indices = scores.argmin(dim=1)

    return {
        "scores": scores,
        "selected_indices": selected_indices,
        "selected_trajectories": _gather_candidates(predicted_samples, selected_indices),
        "route_error": route_error,
        "progress": progress,
        "min_clearance": min_clearance,
        "collision": collision,
        "comfort_penalty": comfort_penalty,
    }


@torch.no_grad()
def select_best_trajectory_samples(
    predicted_samples: torch.Tensor,
    scene_batch: CanonicalSceneBatch,
    time_delta: float = 1.0 / 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one trajectory per scene from a candidate bundle."""

    scored = score_trajectory_candidates(
        predicted_samples=predicted_samples,
        scene_batch=scene_batch,
        time_delta=time_delta,
    )
    return (
        scored["selected_trajectories"],
        scored["selected_indices"],
        scored["scores"],
    )
