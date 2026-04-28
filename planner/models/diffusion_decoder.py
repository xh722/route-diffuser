"""Conditional 1D U-Net used for diffusion trajectory noise prediction."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _resolve_group_count(channels: int, requested_groups: int) -> int:
    groups = min(channels, requested_groups)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for diffusion timesteps."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        ) / max(half_dim - 1, 1)
        frequencies = torch.exp(exponent)
        angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, embedding.new_zeros((embedding.shape[0], 1))], dim=-1
            )
        return embedding


class Conv1dBlock(nn.Module):
    """Conv1d + GroupNorm + SiLU block with same-length padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(_resolve_group_count(out_channels, n_groups), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual 1D block modulated by timestep and scene context."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int,
        n_groups: int,
    ) -> None:
        super().__init__()
        self.block1 = Conv1dBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        self.block2 = Conv1dBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        self.cond_encoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        scale, bias = self.cond_encoder(cond).chunk(2, dim=-1)
        out = out * (1.0 + scale.unsqueeze(-1)) + bias.unsqueeze(-1)
        out = self.block2(out)
        return out + self.residual(x)


class Downsample1d(nn.Module):
    """Stride-2 convolution that halves the sequence length."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layer = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Upsample1d(nn.Module):
    """Nearest-neighbor upsampling followed by convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layer = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        x = F.interpolate(x, size=target_length, mode="nearest")
        return self.layer(x)


class DiffusionDecoder(nn.Module):
    """Predict trajectory noise from noisy trajectories and scene context."""

    def __init__(
        self,
        trajectory_dim: int,
        context_dim: int,
        hidden_dim: int,
        time_dim: int,
        down_dims: tuple[int, ...] = (128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        if len(down_dims) < 2:
            raise ValueError("down_dims must contain at least two levels for the U-Net")

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        cond_dim = hidden_dim + context_dim
        all_dims = [trajectory_dim, *down_dims]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        start_dim = down_dims[0]
        mid_dim = down_dims[-1]

        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            in_channels=dim_in,
                            out_channels=dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            in_channels=dim_out,
                            out_channels=dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        nn.Identity() if is_last else Downsample1d(dim_out),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    in_channels=mid_dim,
                    out_channels=mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    in_channels=mid_dim,
                    out_channels=mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        self.up_modules = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            in_channels=dim_out * 2,
                            out_channels=dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(
                in_channels=start_dim,
                out_channels=start_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
            ),
            nn.Conv1d(start_dim, trajectory_dim, kernel_size=1),
        )

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        sample = noisy_trajectory.moveaxis(-1, -2)
        time_feature = self.time_projection(self.time_embedding(timesteps))
        cond = torch.cat([time_feature, context], dim=-1)

        x = sample
        skips: list[torch.Tensor] = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            skips.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, cond)

        original_length = sample.shape[-1]
        for index, (resnet1, resnet2, upsample) in enumerate(self.up_modules):
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = resnet1(x, cond)
            x = resnet2(x, cond)

            target_length = (
                skips[-1].shape[-1] if index < len(self.up_modules) - 1 else original_length
            )
            x = upsample(x, target_length=target_length)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)
