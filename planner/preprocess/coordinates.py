"""Coordinate frame conversion helpers."""

from __future__ import annotations

import torch


def rotation_matrix(yaw: torch.Tensor) -> torch.Tensor:
    """Create a 2D rotation matrix from yaw angles."""

    yaw = torch.as_tensor(yaw)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    return torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw], dim=-1),
            torch.stack([sin_yaw, cos_yaw], dim=-1),
        ],
        dim=-2,
    )


def global_to_local(
    points_xy: torch.Tensor, origin_xy: torch.Tensor, origin_yaw: torch.Tensor
) -> torch.Tensor:
    """Transform world-frame points into the local ego frame."""

    translated = points_xy - origin_xy.unsqueeze(-2)
    rot = rotation_matrix(-origin_yaw)
    return torch.matmul(translated, rot.transpose(-1, -2))


def local_to_global(
    points_xy: torch.Tensor, origin_xy: torch.Tensor, origin_yaw: torch.Tensor
) -> torch.Tensor:
    """Transform local ego-frame points back into world coordinates."""

    rot = rotation_matrix(origin_yaw)
    rotated = torch.matmul(points_xy, rot.transpose(-1, -2))
    return rotated + origin_xy.unsqueeze(-2)
