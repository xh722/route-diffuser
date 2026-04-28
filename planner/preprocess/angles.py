"""Angle conversion helpers."""

from __future__ import annotations

import torch


def heading_to_cos_sin(heading: torch.Tensor) -> torch.Tensor:
    """Convert heading angles to continuous cosine and sine features."""

    heading = torch.as_tensor(heading)
    return torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)


def cos_sin_to_heading(cos_sin: torch.Tensor) -> torch.Tensor:
    """Convert cosine/sine heading features back to angles."""

    cos_sin = torch.as_tensor(cos_sin)
    if cos_sin.shape[-1] != 2:
        raise ValueError("Expected cosine/sine features in the last dimension")
    return torch.atan2(cos_sin[..., 1], cos_sin[..., 0])
