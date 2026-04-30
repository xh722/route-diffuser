"""Offline diffusion planner MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from planner.datasets.schema import CanonicalSceneBatch
from planner.diffusion.noise import sample_diffusion_noise
from planner.diffusion.schedule import DiffusionSchedule
from planner.diffusion.utils import ddpm_step, q_sample
from planner.inference.anchoring import anchor_first_timestep
from planner.losses.diffusion import (
    candidate_score_distillation_loss,
    noise_prediction_loss,
)
from planner.models.diffusion_decoder import DiffusionDecoder
from planner.models.scene_encoder import SceneEncoder
from planner.models.trajectory_scorer import TrajectoryScorer
from planner.preprocess import build_route_trajectory_prior


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
    route_query_step: float = 2.5
    use_route_prior: bool = True
    diffusion_noise_mode: str = "pyramid"
    diffusion_noise_discount: float = 0.9
    diffusion_steps: int = 32
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    scorer_hidden_dim: int = 128
    learned_scorer_weight: float = 0.0
    scorer_num_candidates: int = 4
    scorer_candidate_noise_scale: float = 0.5
    scorer_target_temperature: float = 0.5
    scene_fusion_mode: str = "concat_mlp"
    scene_attention_heads: int = 4
    scene_attention_layers: int = 1

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
            fusion_mode=config.scene_fusion_mode,
            attention_heads=config.scene_attention_heads,
            attention_layers=config.scene_attention_layers,
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
        self.scorer = TrajectoryScorer(
            context_dim=config.hidden_dim,
            trajectory_dim=config.trajectory_dim,
            hidden_dim=config.scorer_hidden_dim,
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
        context = self.encode_scene(scene_batch)
        return self.decoder(noisy_trajectory, timesteps, context)

    def encode_scene(self, scene_batch: CanonicalSceneBatch) -> torch.Tensor:
        scene_batch.validate()
        return self.encoder(scene_batch)

    def score_trajectories(
        self,
        scene_batch: CanonicalSceneBatch,
        candidate_trajectories: torch.Tensor,
    ) -> torch.Tensor:
        """Return learned preference logits for candidate trajectories."""

        scene_context = self.encode_scene(scene_batch)
        return self.scorer(candidate_trajectories, scene_context)

    def build_trajectory_prior(self, scene_batch: CanonicalSceneBatch) -> torch.Tensor:
        if not self.config.use_route_prior:
            prior = torch.zeros(
                scene_batch.batch_size,
                self.config.future_horizon,
                self.config.trajectory_dim,
                device=scene_batch.ego_current_state.device,
                dtype=scene_batch.ego_current_state.dtype,
            )
            prior[:, 0] = scene_batch.ego_current_state
            return prior

        return build_route_trajectory_prior(
            scene_batch=scene_batch,
            future_horizon=self.config.future_horizon,
            longitudinal_step=self.config.route_query_step,
        )

    def _training_mask(self, scene_batch: CanonicalSceneBatch) -> torch.Tensor:
        mask = (
            scene_batch.future_ego_mask
            if scene_batch.future_ego_mask is not None
            else torch.ones(
                scene_batch.batch_size,
                self.config.future_horizon,
                device=scene_batch.ego_current_state.device,
                dtype=torch.bool,
            )
        )
        mask = mask.clone()
        mask[:, 0] = False
        return mask

    def _build_scorer_candidate_set(
        self,
        scene_batch: CanonicalSceneBatch,
        target_trajectory: torch.Tensor,
        trajectory_prior: torch.Tensor,
    ) -> torch.Tensor:
        num_candidates = max(int(self.config.scorer_num_candidates), 2)
        batch_size, horizon, trajectory_dim = target_trajectory.shape
        device = target_trajectory.device
        dtype = target_trajectory.dtype

        candidates = torch.zeros(
            batch_size,
            num_candidates,
            horizon,
            trajectory_dim,
            device=device,
            dtype=dtype,
        )
        candidates[:, 0] = target_trajectory
        candidates[:, 1] = trajectory_prior

        if num_candidates > 2:
            noise = torch.randn(
                batch_size,
                num_candidates - 2,
                horizon,
                trajectory_dim,
                device=device,
                dtype=dtype,
            )
            noise[:, :, 0] = 0.0
            perturbed = target_trajectory.unsqueeze(1) + self.config.scorer_candidate_noise_scale * noise
            perturbed[:, :, 0] = scene_batch.ego_current_state.unsqueeze(1)
            candidates[:, 2:] = perturbed

        candidates[:, :, 0] = scene_batch.ego_current_state.unsqueeze(1)
        return candidates

    def _candidate_target_costs(
        self,
        candidate_trajectories: torch.Tensor,
        target_trajectory: torch.Tensor,
        future_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        errors = torch.linalg.norm(
            candidate_trajectories[..., :2] - target_trajectory.unsqueeze(1)[..., :2],
            dim=-1,
        )
        if future_mask is None:
            return errors.mean(dim=-1)

        weights = future_mask.to(errors.dtype).unsqueeze(1)
        return (errors * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1.0)

    def training_loss(
        self, scene_batch: CanonicalSceneBatch, noise: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        scene_batch.validate()
        if scene_batch.future_ego_trajectory is None:
            raise ValueError("future_ego_trajectory is required for training")

        prior = self.build_trajectory_prior(scene_batch)
        target = scene_batch.future_ego_trajectory - prior
        batch_size = target.shape[0]
        timesteps = torch.randint(
            low=0,
            high=self.config.diffusion_steps,
            size=(batch_size,),
            device=target.device,
            dtype=torch.long,
        )
        if noise is None:
            noise = sample_diffusion_noise(
                target,
                mode=self.config.diffusion_noise_mode,
                pyramid_discount=self.config.diffusion_noise_discount,
            )
        noise = noise.clone()
        noise[:, 0] = 0.0
        noisy_target = q_sample(target, timesteps, self.schedule, noise=noise)
        noisy_target[:, 0] = 0.0
        pred_noise = self(scene_batch, noisy_target, timesteps)
        diffusion_loss = noise_prediction_loss(
            pred_noise,
            noise,
            mask=self._training_mask(scene_batch),
        )
        scorer_loss = target.new_zeros(())
        scorer_accuracy = target.new_zeros(())

        if self.config.learned_scorer_weight > 0.0:
            candidate_trajectories = self._build_scorer_candidate_set(
                scene_batch=scene_batch,
                target_trajectory=scene_batch.future_ego_trajectory,
                trajectory_prior=prior,
            )
            predicted_scores = self.score_trajectories(scene_batch, candidate_trajectories)
            target_costs = self._candidate_target_costs(
                candidate_trajectories=candidate_trajectories,
                target_trajectory=scene_batch.future_ego_trajectory,
                future_mask=scene_batch.future_ego_mask,
            )
            scorer_loss = candidate_score_distillation_loss(
                predicted_scores,
                target_costs,
                temperature=self.config.scorer_target_temperature,
            )
            scorer_accuracy = (
                predicted_scores.argmax(dim=1) == target_costs.argmin(dim=1)
            ).to(target.dtype).mean()

        loss = diffusion_loss + self.config.learned_scorer_weight * scorer_loss
        return {
            "loss": loss,
            "diffusion_loss": diffusion_loss,
            "scorer_loss": scorer_loss,
            "scorer_accuracy": scorer_accuracy,
            "pred_noise": pred_noise,
            "target_noise": noise,
            "trajectory_prior": prior,
            "timesteps": timesteps,
        }

    @torch.no_grad()
    def sample(
        self, scene_batch: CanonicalSceneBatch, num_samples: int = 1
    ) -> torch.Tensor:
        scene_batch.validate()
        batch_size = scene_batch.batch_size
        device = scene_batch.ego_current_state.device
        prior = self.build_trajectory_prior(scene_batch)
        outputs = []

        for _ in range(num_samples):
            residual = torch.randn(
                batch_size,
                self.config.future_horizon,
                self.config.trajectory_dim,
                device=device,
            )
            residual[:, 0] = 0.0

            for step in reversed(range(self.config.diffusion_steps)):
                timesteps = torch.full(
                    (batch_size,), step, device=device, dtype=torch.long
                )
                pred_noise = self(scene_batch, residual, timesteps)
                residual = ddpm_step(residual, pred_noise, timesteps, self.schedule)
                residual[:, 0] = 0.0

            sample = prior + residual
            sample = anchor_first_timestep(sample, scene_batch.ego_current_state)
            outputs.append(sample)

        return torch.stack(outputs, dim=1)
