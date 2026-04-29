"""Learned candidate trajectory scorer for planner-side ranking experiments."""

from __future__ import annotations

import torch
from torch import nn


class TrajectoryScorer(nn.Module):
    """Score candidate trajectories conditioned on a global scene context."""

    def __init__(
        self,
        *,
        context_dim: int,
        trajectory_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.step_projection = nn.Linear(trajectory_dim, hidden_dim)
        self.step_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        candidate_feature_dim = hidden_dim + trajectory_dim * 2 + 2
        self.head = nn.Sequential(
            nn.LayerNorm(context_dim + candidate_feature_dim),
            nn.Linear(context_dim + candidate_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        candidate_trajectories: torch.Tensor,
        scene_context: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_trajectories.ndim != 4:
            raise ValueError("candidate_trajectories must have shape [B, S, T, D]")
        if scene_context.ndim != 2:
            raise ValueError("scene_context must have shape [B, C]")
        if candidate_trajectories.shape[0] != scene_context.shape[0]:
            raise ValueError("Batch size mismatch between trajectories and scene context")

        batch_size, num_samples, _, trajectory_dim = candidate_trajectories.shape
        flattened = candidate_trajectories.reshape(batch_size * num_samples, -1, trajectory_dim)
        step_features = self.step_projection(flattened)
        step_features = step_features + self.step_mlp(step_features)
        pooled = step_features.mean(dim=1)

        start_state = flattened[:, 0]
        final_state = flattened[:, -1]
        displacement = final_state[:, :2] - start_state[:, :2]
        candidate_features = torch.cat(
            [pooled, start_state, final_state, displacement],
            dim=-1,
        )

        repeated_context = scene_context.unsqueeze(1).expand(-1, num_samples, -1).reshape(
            batch_size * num_samples, -1
        )
        fused = torch.cat([repeated_context, candidate_features], dim=-1)
        logits = self.head(fused).reshape(batch_size, num_samples)
        return logits
