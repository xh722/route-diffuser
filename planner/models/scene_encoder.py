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


class AttentionFusionBlock(nn.Module):
    """Fuse modality tokens with lightweight self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class SceneEncoder(nn.Module):
    """Encode planner scene tensors into a single global context vector."""

    def __init__(
        self,
        ego_dim: int,
        neighbor_dim: int,
        lane_dim: int,
        hidden_dim: int,
        fusion_mode: str = "concat_mlp",
        attention_heads: int = 4,
        attention_layers: int = 1,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"concat_mlp", "token_attention"}:
            raise ValueError(
                f"Unsupported fusion_mode {fusion_mode!r}; expected concat_mlp or token_attention"
            )
        if attention_layers <= 0:
            raise ValueError("attention_layers must be positive")

        self.fusion_mode = fusion_mode
        self.ego_projection = nn.Sequential(
            nn.Linear(ego_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.neighbor_encoder = SetEncoder(input_dim=neighbor_dim, hidden_dim=hidden_dim)
        self.lane_encoder = SetEncoder(input_dim=lane_dim, hidden_dim=hidden_dim)
        self.route_encoder = SetEncoder(input_dim=lane_dim, hidden_dim=hidden_dim)
        self.concat_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.modality_embeddings = nn.Parameter(torch.randn(4, hidden_dim) * 0.02)
        self.attention_blocks = nn.ModuleList(
            [
                AttentionFusionBlock(hidden_dim=hidden_dim, num_heads=attention_heads)
                for _ in range(attention_layers)
            ]
        )
        self.attention_output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        if self.fusion_mode == "concat_mlp":
            fused = torch.cat(
                [ego_feature, neighbor_feature, lane_feature, route_feature],
                dim=-1,
            )
            return self.concat_fusion(fused)

        modality_tokens = torch.stack(
            [ego_feature, neighbor_feature, lane_feature, route_feature],
            dim=1,
        )
        modality_tokens = modality_tokens + self.modality_embeddings.unsqueeze(0)
        for block in self.attention_blocks:
            modality_tokens = block(modality_tokens)

        ego_token = modality_tokens[:, 0]
        pooled_token = modality_tokens.mean(dim=1)
        fused = torch.cat([ego_token, pooled_token], dim=-1)
        return self.attention_output(fused)
