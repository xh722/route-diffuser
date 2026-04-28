"""Inference-time physical constraints."""

from __future__ import annotations

import torch


def anchor_first_timestep(
    trajectories: torch.Tensor, current_state: torch.Tensor
) -> torch.Tensor:
    """Force the first predicted state to match the current ego state."""

    if trajectories.shape[0] != current_state.shape[0]:
        raise ValueError("Batch size mismatch between trajectories and current_state")
    if trajectories.shape[-1] != current_state.shape[-1]:
        raise ValueError("State dimension mismatch between trajectories and current_state")

    anchored = trajectories.clone()
    anchored[:, 0] = current_state
    return anchored
