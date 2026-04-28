"""Diffusion noise schedule definitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed coefficients for a DDPM-style diffusion process."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @classmethod
    def linear(
        cls,
        num_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str | None = None,
    ) -> "DiffusionSchedule":
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return cls(betas=betas, alphas=alphas, alpha_bars=alpha_bars)
