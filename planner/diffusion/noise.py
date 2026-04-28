"""Diffusion noise generation utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pyramid_noise_like(trajectory: torch.Tensor, discount: float = 0.9) -> torch.Tensor:
    """Generate multi-resolution noise following the reference planner design."""

    batch_size, horizon, channels = trajectory.shape
    noise = torch.randn(batch_size, channels, horizon, device=trajectory.device, dtype=trajectory.dtype)
    for level in range(10):
        ratio = torch.rand(1, device=trajectory.device, dtype=trajectory.dtype) + 1.0
        resolution = max(1, int(horizon / (ratio.item() ** level)))
        residual = torch.randn(
            batch_size,
            channels,
            resolution,
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        residual = F.interpolate(residual, size=horizon, mode="linear", align_corners=False)
        noise = noise + residual * (discount**level)
        if resolution == 1:
            break
    noise = noise.permute(0, 2, 1)
    return noise / noise.std().clamp(min=1e-6)


def sample_diffusion_noise(
    like: torch.Tensor,
    mode: str = "gaussian",
    pyramid_discount: float = 0.9,
) -> torch.Tensor:
    """Sample diffusion training noise according to the configured mode."""

    if mode == "gaussian":
        return torch.randn_like(like)
    if mode == "pyramid":
        return pyramid_noise_like(like, discount=pyramid_discount)
    raise ValueError(f"Unsupported diffusion noise mode: {mode}")
