"""Scene encoder boundary for dynamic agents, lanes, and route context."""

from __future__ import annotations

import torch
from torch import nn

from planner.datasets.schema import CanonicalSceneBatch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute a masked mean over all token dimensions except batch and channel."""

    if values.ndim != mask.ndim + 1:
        raise ValueError("values rank must equal mask rank + 1")
    flat_values = values.reshape(values.shape[0], -1, values.shape[-1])
    flat_mask = mask.reshape(mask.shape[0], -1).to(values.dtype).unsqueeze(-1)
    numerator = (flat_values * flat_mask).sum(dim=1)
    denominator = flat_mask.sum(dim=1).clamp(min=1.0)
    return numerator / denominator


class SceneEncoder(nn.Module):
    """Encode planner scene tensors into a single global context vector."""

    def __init__(
        self,
        ego_dim: int,
        neighbor_dim: int,
        lane_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.ego_projection = nn.Linear(ego_dim, hidden_dim)
        self.neighbor_projection = nn.Linear(neighbor_dim, hidden_dim)
        self.lane_projection = nn.Linear(lane_dim, hidden_dim)
        self.route_projection = nn.Linear(lane_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, scene_batch: CanonicalSceneBatch) -> torch.Tensor:
        scene_batch.validate()
        ego_feature = self.ego_projection(scene_batch.ego_current_state)
        neighbor_feature = self.neighbor_projection(
            masked_mean(scene_batch.neighbor_history, scene_batch.neighbor_history_mask)
        )
        lane_feature = self.lane_projection(
            masked_mean(scene_batch.lane_polylines, scene_batch.lane_polylines_mask)
        )
        route_feature = self.route_projection(
            masked_mean(scene_batch.route_lanes, scene_batch.route_lanes_mask)
        )
        fused = torch.cat(
            [ego_feature, neighbor_feature, lane_feature, route_feature], dim=-1
        )
        return self.fusion(fused)
