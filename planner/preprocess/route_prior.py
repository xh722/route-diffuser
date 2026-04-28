"""Route-conditioned trajectory priors for planner training and inference."""

from __future__ import annotations

import torch

from planner.datasets.schema import CanonicalSceneBatch


def build_route_trajectory_prior(
    scene_batch: CanonicalSceneBatch,
    future_horizon: int,
    longitudinal_step: float,
) -> torch.Tensor:
    """Construct a simple route-following trajectory prior in the planner frame."""

    scene_batch.validate()
    device = scene_batch.ego_current_state.device
    dtype = scene_batch.ego_current_state.dtype

    query_x = torch.arange(future_horizon, device=device, dtype=dtype) * longitudinal_step
    priors = []
    for batch_index in range(scene_batch.batch_size):
        route_polylines = scene_batch.route_lanes[batch_index]
        route_mask = scene_batch.route_lanes_mask[batch_index]
        valid_lane_index = _first_valid_lane_index(route_mask)
        lane_points = route_polylines[valid_lane_index][route_mask[valid_lane_index]]
        if lane_points.shape[0] == 0:
            raise ValueError("route prior requires at least one valid route point")
        priors.append(
            _interpolate_route_states(
                lane_points=lane_points,
                current_state=scene_batch.ego_current_state[batch_index],
                query_x=query_x,
            )
        )

    return torch.stack(priors, dim=0)


def _first_valid_lane_index(route_mask: torch.Tensor) -> int:
    valid_lanes = route_mask.any(dim=-1)
    indices = torch.nonzero(valid_lanes, as_tuple=False)
    if indices.numel() == 0:
        return 0
    return int(indices[0].item())


def _interpolate_route_states(
    lane_points: torch.Tensor,
    current_state: torch.Tensor,
    query_x: torch.Tensor,
) -> torch.Tensor:
    trajectory_dim = int(current_state.shape[-1])
    prior = torch.zeros(query_x.shape[0], trajectory_dim, device=lane_points.device, dtype=lane_points.dtype)

    x_src = lane_points[:, 0]
    y_src = lane_points[:, 1]
    cos_src = lane_points[:, 2]
    sin_src = lane_points[:, 3]
    x_src = x_src.contiguous()

    clamped_x = query_x.clamp(min=float(x_src[0].item()), max=float(x_src[-1].item()))
    upper = torch.searchsorted(x_src, clamped_x, right=True).clamp(
        min=1, max=x_src.shape[0] - 1
    )
    lower = upper - 1
    denom = (x_src[upper] - x_src[lower]).clamp(min=1e-6)
    weight = (clamped_x - x_src[lower]) / denom

    prior[:, 0] = clamped_x
    prior[:, 1] = torch.lerp(y_src[lower], y_src[upper], weight)
    prior[:, 2] = torch.lerp(cos_src[lower], cos_src[upper], weight)
    prior[:, 3] = torch.lerp(sin_src[lower], sin_src[upper], weight)

    heading_norm = torch.linalg.norm(prior[:, 2:4], dim=-1, keepdim=True).clamp(min=1e-6)
    prior[:, 2:4] = prior[:, 2:4] / heading_norm

    speed = torch.linalg.norm(current_state[4:6]).clamp(min=1e-6)
    prior[:, 4:6] = speed * prior[:, 2:4]
    prior[0] = current_state
    return prior
