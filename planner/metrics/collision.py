"""Oriented-box collision checks for planner metrics."""

from __future__ import annotations

import torch


def _state_heading(states: torch.Tensor) -> torch.Tensor:
    if states.shape[-1] >= 4:
        return torch.atan2(states[..., 3], states[..., 2])
    return states.new_zeros(states.shape[:-1])


def oriented_box_corners(
    states: torch.Tensor,
    *,
    length: float,
    width: float,
) -> torch.Tensor:
    """Build box corners from trajectory states with `[x, y, cos, sin, ...]` layout."""

    if states.shape[-1] < 2:
        raise ValueError("states must include at least x and y coordinates")
    if length <= 0.0 or width <= 0.0:
        raise ValueError("length and width must be positive")

    heading = _state_heading(states)
    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)
    forward = torch.stack([cos_heading, sin_heading], dim=-1) * (length * 0.5)
    lateral = torch.stack([-sin_heading, cos_heading], dim=-1) * (width * 0.5)
    center = states[..., :2]
    return torch.stack(
        [
            center + forward + lateral,
            center + forward - lateral,
            center - forward - lateral,
            center - forward + lateral,
        ],
        dim=-2,
    )


def _normalize_axes(axes: torch.Tensor) -> torch.Tensor:
    return axes / torch.linalg.norm(axes, dim=-1, keepdim=True).clamp(min=1e-6)


def box_collision_matrix(
    ego_states: torch.Tensor,
    object_states: torch.Tensor,
    *,
    ego_length: float = 4.8,
    ego_width: float = 2.0,
    object_length: float = 4.8,
    object_width: float = 2.0,
) -> torch.Tensor:
    """Check oriented-box overlap with the Separating Axis Theorem.

    Args:
        ego_states: `[B, T, D]` ego states.
        object_states: `[B, N, T, D]` object states aligned to the same horizon.

    Returns:
        Boolean collision matrix with shape `[B, N, T]`.
    """

    if ego_states.ndim != 3:
        raise ValueError("ego_states must have shape [B, T, D]")
    if object_states.ndim != 4:
        raise ValueError("object_states must have shape [B, N, T, D]")
    if ego_states.shape[0] != object_states.shape[0]:
        raise ValueError("ego_states and object_states batch dimensions must match")
    if ego_states.shape[1] != object_states.shape[2]:
        raise ValueError("ego_states and object_states horizon dimensions must match")

    ego_corners = oriented_box_corners(
        ego_states,
        length=ego_length,
        width=ego_width,
    ).unsqueeze(1)
    object_corners = oriented_box_corners(
        object_states,
        length=object_length,
        width=object_width,
    )

    ego_axes = _normalize_axes(
        torch.stack(
            [
                ego_corners[..., 1, :] - ego_corners[..., 0, :],
                ego_corners[..., 3, :] - ego_corners[..., 0, :],
            ],
            dim=-2,
        )
    ).expand(-1, object_states.shape[1], -1, -1, -1)
    object_axes = _normalize_axes(
        torch.stack(
            [
                object_corners[..., 1, :] - object_corners[..., 0, :],
                object_corners[..., 3, :] - object_corners[..., 0, :],
            ],
            dim=-2,
        )
    )
    axes = torch.cat([ego_axes, object_axes], dim=-2)
    ego_projection = (ego_corners.unsqueeze(-3) * axes.unsqueeze(-2)).sum(dim=-1)
    object_projection = (
        object_corners.unsqueeze(-3) * axes.unsqueeze(-2)
    ).sum(dim=-1)

    separated = (
        ego_projection.amax(dim=-1) < object_projection.amin(dim=-1)
    ) | (
        object_projection.amax(dim=-1) < ego_projection.amin(dim=-1)
    )
    return ~separated.any(dim=-1)

