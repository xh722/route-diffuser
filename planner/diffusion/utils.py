"""Core diffusion tensor operations."""

from __future__ import annotations

import torch

from planner.diffusion.schedule import DiffusionSchedule


def extract(values: torch.Tensor, timesteps: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Gather timestep-aligned coefficients and reshape them for broadcasting."""

    gathered = values.index_select(0, timesteps)
    return gathered.view(like.shape[0], *([1] * (like.ndim - 1)))


def q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward diffuse a clean trajectory sample."""

    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bar_t = extract(schedule.alpha_bars, timesteps, x0)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise


def ddpm_step(
    x_t: torch.Tensor,
    pred_noise: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one reverse-diffusion DDPM update step."""

    if noise is None:
        noise = torch.randn_like(x_t)

    beta_t = extract(schedule.betas, timesteps, x_t)
    alpha_t = extract(schedule.alphas, timesteps, x_t)
    alpha_bar_t = extract(schedule.alpha_bars, timesteps, x_t)
    mean = (x_t - beta_t * pred_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)

    nonzero = (timesteps > 0).to(x_t.dtype).view(x_t.shape[0], *([1] * (x_t.ndim - 1)))
    return mean + nonzero * torch.sqrt(beta_t) * noise
