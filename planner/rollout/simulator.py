"""Lightweight closed-loop rollout utilities for RouteDiffuser."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from planner.datasets.schema import CanonicalSceneBatch
from planner.inference import score_trajectory_candidates_for_mode
from planner.metrics import box_collision_matrix
from planner.preprocess import cos_sin_to_heading, heading_to_cos_sin


@dataclass
class RolloutResult:
    """Closed-loop rollout trace and summary metadata."""

    executed_world_states: torch.Tensor
    reference_world_states: torch.Tensor | None
    route_world_polylines: torch.Tensor
    selected_indices: torch.Tensor
    selected_scores: torch.Tensor
    collision_flags: torch.Tensor
    point_collision_flags: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


def _rotate_vectors(vectors: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    x = vectors[..., 0]
    y = vectors[..., 1]
    return torch.stack(
        [
            x * cos_yaw - y * sin_yaw,
            x * sin_yaw + y * cos_yaw,
        ],
        dim=-1,
    )


def _state_local_to_world(
    states: torch.Tensor,
    frame_origin_xy: torch.Tensor,
    frame_yaw: torch.Tensor,
) -> torch.Tensor:
    xy_world = _rotate_vectors(states[..., :2], frame_yaw) + frame_origin_xy
    heading_world = cos_sin_to_heading(states[..., 2:4]) + frame_yaw
    velocity_world = _rotate_vectors(states[..., 4:6], frame_yaw)

    world = states.clone()
    world[..., :2] = xy_world
    world[..., 2:4] = heading_to_cos_sin(heading_world)
    world[..., 4:6] = velocity_world
    return world


def _state_world_to_local(
    states: torch.Tensor,
    frame_origin_xy: torch.Tensor,
    frame_yaw: torch.Tensor,
) -> torch.Tensor:
    translated_xy = states[..., :2] - frame_origin_xy
    xy_local = _rotate_vectors(translated_xy, -frame_yaw)
    heading_local = cos_sin_to_heading(states[..., 2:4]) - frame_yaw
    velocity_local = _rotate_vectors(states[..., 4:6], -frame_yaw)

    local = states.clone()
    local[..., :2] = xy_local
    local[..., 2:4] = heading_to_cos_sin(heading_local)
    local[..., 4:6] = velocity_local
    return local


def _polyline_local_to_world(
    polylines: torch.Tensor,
    frame_origin_xy: torch.Tensor,
    frame_yaw: torch.Tensor,
) -> torch.Tensor:
    xy_world = _rotate_vectors(polylines[..., :2], frame_yaw) + frame_origin_xy
    heading_world = cos_sin_to_heading(polylines[..., 2:4]) + frame_yaw

    world = polylines.clone()
    world[..., :2] = xy_world
    world[..., 2:4] = heading_to_cos_sin(heading_world)
    return world


def _polyline_world_to_local(
    polylines: torch.Tensor,
    frame_origin_xy: torch.Tensor,
    frame_yaw: torch.Tensor,
) -> torch.Tensor:
    translated_xy = polylines[..., :2] - frame_origin_xy
    xy_local = _rotate_vectors(translated_xy, -frame_yaw)
    heading_local = cos_sin_to_heading(polylines[..., 2:4]) - frame_yaw

    local = polylines.clone()
    local[..., :2] = xy_local
    local[..., 2:4] = heading_to_cos_sin(heading_local)
    return local


def _advance_neighbor_history_world(
    history_world: torch.Tensor,
    history_mask: torch.Tensor,
    time_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    updated_history = history_world.clone()
    updated_mask = history_mask.clone()

    for neighbor_index in range(history_world.shape[0]):
        if not history_mask[neighbor_index].any():
            continue
        last_valid = int(torch.nonzero(history_mask[neighbor_index], as_tuple=False)[-1].item())
        last_state = history_world[neighbor_index, last_valid]
        next_state = last_state.clone()
        next_state[:2] = last_state[:2] + last_state[4:6] * time_delta

        updated_history[neighbor_index, :-1] = history_world[neighbor_index, 1:]
        updated_history[neighbor_index, -1] = next_state
        updated_mask[neighbor_index, :-1] = history_mask[neighbor_index, 1:]
        updated_mask[neighbor_index, -1] = history_mask[neighbor_index].any()

    return updated_history, updated_mask


def _route_error_world(
    executed_world_states: torch.Tensor,
    route_world_polylines: torch.Tensor,
    route_world_mask: torch.Tensor,
) -> float:
    route_points = route_world_polylines[..., :2][route_world_mask]
    if route_points.numel() == 0:
        return float("nan")
    distances = torch.cdist(executed_world_states[..., :2], route_points.unsqueeze(0)).squeeze(0)
    return float(distances.min(dim=-1).values.mean().item())


def _collision_flags_world(
    executed_world_states: torch.Tensor,
    neighbor_world_states: list[torch.Tensor],
    neighbor_world_masks: list[torch.Tensor],
    threshold: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not neighbor_world_states:
        empty = torch.zeros(executed_world_states.shape[0] - 1, dtype=torch.bool)
        return empty, empty

    box_flags = []
    point_flags = []
    for step_index, ego_state in enumerate(executed_world_states[1:]):
        neighbor_states = neighbor_world_states[step_index]
        neighbor_mask = neighbor_world_masks[step_index]
        neighbors = neighbor_states[..., :2]
        distances = torch.linalg.norm(neighbors - ego_state[:2], dim=-1)
        point_flags.append(bool(((distances < threshold) & neighbor_mask).any().item()))
        box_collision = box_collision_matrix(
            ego_state.view(1, 1, -1),
            neighbor_states.view(1, neighbor_states.shape[0], 1, -1),
        ).view(-1)
        box_flags.append(bool((box_collision & neighbor_mask).any().item()))
    return torch.tensor(box_flags, dtype=torch.bool), torch.tensor(point_flags, dtype=torch.bool)


def summarize_rollout(result: RolloutResult) -> dict[str, float | int | str]:
    """Summarize a closed-loop rollout into lightweight scalar metrics."""

    executed = result.executed_world_states
    reference = result.reference_world_states

    summary: dict[str, float | int | str] = {
        "num_steps": int(executed.shape[0] - 1),
        "scenario_name": str(result.metadata.get("scenario_name", "unknown")),
        "route_error": _route_error_world(
            executed,
            result.route_world_polylines,
            result.metadata["route_world_mask"],
        ),
        "collision_rate": float(result.collision_flags.to(torch.float32).mean().item())
        if result.collision_flags.numel()
        else 0.0,
        "box_collision_rate": float(result.collision_flags.to(torch.float32).mean().item())
        if result.collision_flags.numel()
        else 0.0,
        "point_collision_rate": float(result.point_collision_flags.to(torch.float32).mean().item())
        if result.point_collision_flags.numel()
        else 0.0,
        "mean_selected_index": float(result.selected_indices.to(torch.float32).mean().item())
        if result.selected_indices.numel()
        else 0.0,
        "mean_selected_score": float(result.selected_scores.mean().item())
        if result.selected_scores.numel()
        else 0.0,
    }

    if reference is not None:
        xy_error = torch.linalg.norm(executed[..., :2] - reference[..., :2], dim=-1)
        summary["closed_loop_ade"] = float(xy_error.mean().item())
        summary["closed_loop_fde"] = float(xy_error[-1].item())
    else:
        summary["closed_loop_ade"] = float("nan")
        summary["closed_loop_fde"] = float("nan")

    return summary


@torch.no_grad()
def rollout_planner(
    model,
    scene_batch: CanonicalSceneBatch,
    *,
    num_steps: int,
    num_samples: int,
    time_delta: float,
    selection_mode: str = "auto",
) -> RolloutResult:
    """Run a lightweight receding-horizon rollout for one scene."""

    scene_batch.validate()
    if scene_batch.batch_size != 1:
        raise ValueError("rollout_planner currently supports batch_size == 1")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    current_batch = CanonicalSceneBatch(
        ego_current_state=scene_batch.ego_current_state.clone(),
        neighbor_history=scene_batch.neighbor_history.clone(),
        neighbor_history_mask=scene_batch.neighbor_history_mask.clone(),
        lane_polylines=scene_batch.lane_polylines.clone(),
        lane_polylines_mask=scene_batch.lane_polylines_mask.clone(),
        route_lanes=scene_batch.route_lanes.clone(),
        route_lanes_mask=scene_batch.route_lanes_mask.clone(),
        metadata=scene_batch.metadata,
    ).validate()

    current_frame_origin = torch.zeros(2, device=current_batch.ego_current_state.device)
    current_frame_yaw = torch.zeros((), device=current_batch.ego_current_state.device)
    route_world = _polyline_local_to_world(
        current_batch.route_lanes[0],
        current_frame_origin,
        current_frame_yaw,
    )
    route_world_mask = current_batch.route_lanes_mask[0].clone()

    executed_world_states = [
        _state_local_to_world(
            current_batch.ego_current_state[0],
            current_frame_origin,
            current_frame_yaw,
        )
    ]
    reference_world_remaining = None
    reference_world_states = None
    if scene_batch.future_ego_trajectory is not None:
        reference_world_remaining = _state_local_to_world(
            scene_batch.future_ego_trajectory[0],
            current_frame_origin,
            current_frame_yaw,
        )
        reference_world_states = [reference_world_remaining[0]]

    selected_indices: list[int] = []
    selected_scores: list[float] = []
    neighbor_world_states: list[torch.Tensor] = []
    neighbor_world_masks: list[torch.Tensor] = []

    for _ in range(num_steps):
        predictions = model.sample(current_batch, num_samples=num_samples)
        scored = score_trajectory_candidates_for_mode(
            predicted_samples=predictions,
            scene_batch=current_batch,
            model=model,
            time_delta=time_delta,
            selection_mode=selection_mode,
        )
        selected_local = scored["selected_trajectories"][0]
        next_local_state = selected_local[min(1, selected_local.shape[0] - 1)]
        next_world_state = _state_local_to_world(
            next_local_state,
            current_frame_origin,
            current_frame_yaw,
        )

        selected_indices.append(int(scored["selected_indices"][0].item()))
        selected_scores.append(float(scored["scores"][0, scored["selected_indices"][0]].item()))
        executed_world_states.append(next_world_state)

        if reference_world_remaining is not None and reference_world_states is not None:
            reference_next = reference_world_remaining[min(1, reference_world_remaining.shape[0] - 1)]
            reference_world_states.append(reference_next)
            reference_world_remaining = torch.cat(
                [reference_world_remaining[1:], reference_world_remaining[-1:].clone()],
                dim=0,
            )

        history_world = _state_local_to_world(
            current_batch.neighbor_history[0],
            current_frame_origin,
            current_frame_yaw,
        )
        history_mask = current_batch.neighbor_history_mask[0]
        advanced_history_world, advanced_history_mask = _advance_neighbor_history_world(
            history_world,
            history_mask,
            time_delta,
        )
        neighbor_world_states.append(advanced_history_world[:, -1])
        neighbor_world_masks.append(advanced_history_mask[:, -1])

        lane_world = _polyline_local_to_world(
            current_batch.lane_polylines[0],
            current_frame_origin,
            current_frame_yaw,
        )

        next_frame_origin = next_world_state[:2]
        next_frame_yaw = cos_sin_to_heading(next_world_state[2:4])

        current_batch = CanonicalSceneBatch(
            ego_current_state=_state_world_to_local(
                next_world_state.unsqueeze(0),
                next_frame_origin,
                next_frame_yaw,
            ),
            neighbor_history=_state_world_to_local(
                advanced_history_world,
                next_frame_origin,
                next_frame_yaw,
            ).unsqueeze(0),
            neighbor_history_mask=advanced_history_mask.unsqueeze(0),
            lane_polylines=_polyline_world_to_local(
                lane_world,
                next_frame_origin,
                next_frame_yaw,
            ).unsqueeze(0),
            lane_polylines_mask=current_batch.lane_polylines_mask.clone(),
            route_lanes=_polyline_world_to_local(
                route_world,
                next_frame_origin,
                next_frame_yaw,
            ).unsqueeze(0),
            route_lanes_mask=current_batch.route_lanes_mask.clone(),
            metadata=current_batch.metadata,
        ).validate()
        current_frame_origin = next_frame_origin
        current_frame_yaw = next_frame_yaw

    executed_tensor = torch.stack(executed_world_states, dim=0)
    reference_tensor = (
        None if reference_world_states is None else torch.stack(reference_world_states, dim=0)
    )
    collision_flags, point_collision_flags = _collision_flags_world(
        executed_tensor,
        neighbor_world_states,
        neighbor_world_masks,
    )
    scenario_names = list(scene_batch.metadata.get("scenario_names", []))
    scenario_name = scenario_names[0] if scenario_names else "unknown"

    return RolloutResult(
        executed_world_states=executed_tensor,
        reference_world_states=reference_tensor,
        route_world_polylines=route_world,
        selected_indices=torch.tensor(selected_indices, dtype=torch.long),
        selected_scores=torch.tensor(selected_scores, dtype=executed_tensor.dtype),
        collision_flags=collision_flags,
        point_collision_flags=point_collision_flags,
        metadata={
            "scenario_name": scenario_name,
            "route_world_mask": route_world_mask,
        },
    )
