"""Lightweight trajectory reward objectives for planning experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from planner.datasets.schema import CanonicalSceneBatch
from planner.metrics.trajectory import (
    neighbor_clearance_metrics,
    route_consistency_errors,
    trajectory_comfort_metrics,
    trajectory_progress,
)


@dataclass(frozen=True)
class TrajectoryRewardConfig:
    """Reward weights for lightweight planning objectives."""

    progress_weight: float = 0.10
    route_weight: float = 1.00
    clearance_weight: float = 0.50
    collision_penalty: float = 2.00
    comfort_penalty_weight: float = 0.35
    clearance_cap: float = 3.0


def _repeat_scene_tensor(tensor: torch.Tensor, num_samples: int) -> torch.Tensor:
    expand_shape = [tensor.shape[0], num_samples, *tensor.shape[1:]]
    return tensor.unsqueeze(1).expand(*expand_shape).reshape(-1, *tensor.shape[1:])


def _normalize(values: torch.Tensor) -> torch.Tensor:
    scale = values.abs().amax(dim=1, keepdim=True).clamp(min=1.0)
    return values / scale


def _comfort_penalty(comfort_metrics: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        torch.relu(comfort_metrics["max_forward_accel"] - 2.40) / 2.40
        + torch.relu(-4.05 - comfort_metrics["max_brake"]) / 4.05
        + torch.relu(comfort_metrics["max_abs_lateral_accel"] - 4.89) / 4.89
        + torch.relu(comfort_metrics["max_abs_jerk"] - 4.13) / 4.13
        + torch.relu(comfort_metrics["max_abs_yaw_rate"] - 0.95) / 0.95
    )


def trajectory_rewards(
    candidate_trajectories: torch.Tensor,
    scene_batch: CanonicalSceneBatch,
    *,
    time_delta: float,
    config: TrajectoryRewardConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute lightweight scalar rewards for candidate trajectories."""

    reward_config = TrajectoryRewardConfig() if config is None else config
    scene_batch.validate()

    if candidate_trajectories.ndim == 3:
        candidate_trajectories = candidate_trajectories.unsqueeze(1)
    if candidate_trajectories.ndim != 4:
        raise ValueError("candidate_trajectories must have shape [B, S, T, D]")

    batch_size, num_samples, horizon, state_dim = candidate_trajectories.shape
    flattened = candidate_trajectories.reshape(batch_size * num_samples, horizon, state_dim)

    route_error = route_consistency_errors(
        flattened,
        route_polylines=_repeat_scene_tensor(scene_batch.route_lanes, num_samples),
        route_mask=_repeat_scene_tensor(scene_batch.route_lanes_mask, num_samples),
    ).reshape(batch_size, num_samples)
    progress = trajectory_progress(flattened).reshape(batch_size, num_samples)
    clearance = neighbor_clearance_metrics(
        flattened,
        neighbor_history=_repeat_scene_tensor(scene_batch.neighbor_history, num_samples),
        neighbor_history_mask=_repeat_scene_tensor(scene_batch.neighbor_history_mask, num_samples),
        time_delta=time_delta,
        collision_distance=2.0,
    )
    comfort = trajectory_comfort_metrics(flattened, time_delta=time_delta)

    min_clearance = clearance["min_clearance"].reshape(batch_size, num_samples)
    collision = clearance["collision"].reshape(batch_size, num_samples)
    comfort_penalty = _comfort_penalty(comfort).reshape(batch_size, num_samples)
    clearance_bonus = min_clearance.clamp(max=reward_config.clearance_cap) / reward_config.clearance_cap
    normalized_progress = _normalize(progress)

    reward = (
        reward_config.progress_weight * normalized_progress
        - reward_config.route_weight * route_error
        + reward_config.clearance_weight * clearance_bonus
        - reward_config.collision_penalty * collision.to(route_error.dtype)
        - reward_config.comfort_penalty_weight * comfort_penalty
    )

    return {
        "reward": reward,
        "route_error": route_error,
        "progress": progress,
        "min_clearance": min_clearance,
        "collision": collision,
        "comfort_penalty": comfort_penalty,
    }
