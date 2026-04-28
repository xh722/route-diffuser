"""Scene encoder boundary for dynamic agents, lanes, and route context."""

from __future__ import annotations

import torch
from torch import nn

from planner.datasets.schema import CanonicalSceneBatch


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute a masked mean along the target dimension."""

    weights = mask.to(values.dtype).unsqueeze(-1)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp(min=1.0)
    return numerator / denominator


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute a softmax that ignores invalid tokens."""

    masked_logits = logits.masked_fill(~mask, -1e9)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp(min=1e-6)


class SetEncoder(nn.Module):
    """Encode a `[batch, set, token, dim]` tensor with masked token and set pooling."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.token_projection = nn.Linear(input_dim, hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.entity_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or mask.ndim != 3:
            raise ValueError("SetEncoder expects values [B,N,T,D] and mask [B,N,T]")
        if values.shape[:-1] != mask.shape:
            raise ValueError("values and mask shapes do not align")

        token_feature = self.token_projection(values)
        token_feature = token_feature + self.token_mlp(token_feature)
        entity_feature = masked_mean(token_feature, mask, dim=2)
        entity_feature = entity_feature + self.entity_mlp(entity_feature)

        entity_valid = mask.any(dim=-1)
        attention_logits = self.attention(entity_feature).squeeze(-1)
        attention = masked_softmax(attention_logits, entity_valid, dim=1)
        pooled = (entity_feature * attention.unsqueeze(-1)).sum(dim=1)
        return self.output_projection(pooled)


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
        self.ego_projection = nn.Sequential(
            nn.Linear(ego_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.neighbor_encoder = SetEncoder(input_dim=neighbor_dim, hidden_dim=hidden_dim)
        self.lane_encoder = SetEncoder(input_dim=lane_dim, hidden_dim=hidden_dim)
        self.route_encoder = SetEncoder(input_dim=lane_dim, hidden_dim=hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, scene_batch: CanonicalSceneBatch) -> torch.Tensor:
        scene_batch.validate()
        ego_feature = self.ego_projection(scene_batch.ego_current_state)
        neighbor_feature = self.neighbor_encoder(
            scene_batch.neighbor_history,
            scene_batch.neighbor_history_mask,
        )
        lane_feature = self.lane_encoder(
            scene_batch.lane_polylines,
            scene_batch.lane_polylines_mask,
        )
        route_feature = self.route_encoder(
            scene_batch.route_lanes,
            scene_batch.route_lanes_mask,
        )
        fused = torch.cat(
            [ego_feature, neighbor_feature, lane_feature, route_feature],
            dim=-1,
        )
        return self.fusion(fused)
