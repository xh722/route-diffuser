"""Offline diffusion planner MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from planner.datasets.schema import CanonicalSceneBatch
from planner.diffusion.schedule import DiffusionSchedule
from planner.diffusion.utils import ddpm_step, q_sample
from planner.inference.anchoring import anchor_first_timestep
from planner.models.diffusion_decoder import DiffusionDecoder
from planner.models.scene_encoder import SceneEncoder


@dataclass(frozen=True)
class DiffusionPlannerConfig:
    """Configuration for the diffusion planner MVP."""

    ego_dim: int = 6
    neighbor_dim: int = 6
    lane_dim: int = 4
    trajectory_dim: int = 6
    hidden_dim: int = 128
    time_dim: int = 128
    decoder_down_dims: tuple[int, ...] = (128, 256)
    decoder_kernel_size: int = 5
    decoder_groups: int = 8
    future_horizon: int = 16
    diffusion_steps: int = 32
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "DiffusionPlannerConfig":
        filtered = {key: values[key] for key in cls.__dataclass_fields__ if key in values}
        if "decoder_down_dims" in filtered:
            filtered["decoder_down_dims"] = tuple(filtered["decoder_down_dims"])
        return cls(**filtered)


class DiffusionPlanner(nn.Module):
    """Conditional diffusion planner with a lightweight encoder-decoder backbone."""

    def __init__(self, config: DiffusionPlannerConfig):
        super().__init__()
        if config.ego_dim != config.trajectory_dim:
            raise ValueError("ego_dim and trajectory_dim must match for anchoring")

        self.config = config
        self.encoder = SceneEncoder(
            ego_dim=config.ego_dim,
            neighbor_dim=config.neighbor_dim,
            lane_dim=config.lane_dim,
            hidden_dim=config.hidden_dim,
        )
        self.decoder = DiffusionDecoder(
            trajectory_dim=config.trajectory_dim,
            context_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            time_dim=config.time_dim,
            down_dims=config.decoder_down_dims,
            kernel_size=config.decoder_kernel_size,
            n_groups=config.decoder_groups,
        )

        schedule = DiffusionSchedule.linear(
            num_steps=config.diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )
        self.register_buffer("betas", schedule.betas)
        self.register_buffer("alphas", schedule.alphas)
        self.register_buffer("alpha_bars", schedule.alpha_bars)

    @property
    def schedule(self) -> DiffusionSchedule:
        return DiffusionSchedule(
            betas=self.betas, alphas=self.alphas, alpha_bars=self.alpha_bars
        )

    def forward(
        self,
        scene_batch: CanonicalSceneBatch,
        noisy_trajectory: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        scene_batch.validate()
        context = self.encoder(scene_batch)
        return self.decoder(noisy_trajectory, timesteps, context)

    def training_loss(
        self, scene_batch: CanonicalSceneBatch, noise: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        scene_batch.validate()
        if scene_batch.future_ego_trajectory is None:
            raise ValueError("future_ego_trajectory is required for training")

        target = scene_batch.future_ego_trajectory
        batch_size = target.shape[0]
        timesteps = torch.randint(
            low=0,
            high=self.config.diffusion_steps,
            size=(batch_size,),
            device=target.device,
            dtype=torch.long,
        )
        if noise is None:
            noise = torch.randn_like(target)
        noisy_target = q_sample(target, timesteps, self.schedule, noise=noise)
        pred_noise = self(scene_batch, noisy_target, timesteps)
        loss = F.mse_loss(pred_noise, noise)
        return {
            "loss": loss,
            "pred_noise": pred_noise,
            "target_noise": noise,
            "timesteps": timesteps,
        }

    @torch.no_grad()
    def sample(
        self, scene_batch: CanonicalSceneBatch, num_samples: int = 1
    ) -> torch.Tensor:
        scene_batch.validate()
        batch_size = scene_batch.batch_size
        device = scene_batch.ego_current_state.device
        outputs = []

        for _ in range(num_samples):
            sample = torch.randn(
                batch_size,
                self.config.future_horizon,
                self.config.trajectory_dim,
                device=device,
            )
            sample = anchor_first_timestep(sample, scene_batch.ego_current_state)

            for step in reversed(range(self.config.diffusion_steps)):
                timesteps = torch.full(
                    (batch_size,), step, device=device, dtype=torch.long
                )
                pred_noise = self(scene_batch, sample, timesteps)
                sample = ddpm_step(sample, pred_noise, timesteps, self.schedule)
                sample = anchor_first_timestep(sample, scene_batch.ego_current_state)

            outputs.append(sample)

        return torch.stack(outputs, dim=1)
