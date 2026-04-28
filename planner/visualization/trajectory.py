"""Plotting helpers for predicted and ground-truth trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_trajectory_comparison(
    predicted: torch.Tensor,
    target: torch.Tensor,
    route_polylines: torch.Tensor | None = None,
    route_mask: torch.Tensor | None = None,
    output_path: str | Path | None = None,
    title: str = "Predicted vs ground-truth trajectory",
):
    """Plot a single predicted and target trajectory pair."""

    predicted = _select_trajectory(predicted).detach().cpu()
    target = _select_trajectory(target).detach().cpu()
    fig, axis = plt.subplots(figsize=(6, 6))

    if route_polylines is not None:
        route_polylines = _select_polyline_bank(route_polylines).detach().cpu()
        route_mask = (
            None if route_mask is None else _select_polyline_mask(route_mask).detach().cpu()
        )
        for polyline_idx in range(route_polylines.shape[0]):
            points = route_polylines[polyline_idx]
            if route_mask is not None:
                valid = route_mask[polyline_idx].bool()
                points = points[valid]
            if points.numel() == 0:
                continue
            axis.plot(points[:, 0], points[:, 1], color="lightgray", linewidth=1.0)

    axis.plot(target[:, 0], target[:, 1], color="forestgreen", label="target")
    axis.plot(predicted[:, 0], predicted[:, 1], color="firebrick", label="predicted")
    axis.scatter(target[0, 0], target[0, 1], color="black", s=30, label="start")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.axis("equal")
    axis.legend()
    axis.grid(True, linestyle="--", linewidth=0.5)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
        plt.close(fig)
        return destination

    return fig


def _select_trajectory(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor[0, 0]
    elif tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Expected trajectory tensor with rank 2, got {tensor.ndim}")
    return tensor


def _select_polyline_bank(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected polyline bank tensor with rank 3, got {tensor.ndim}")
    return tensor


def _select_polyline_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Expected polyline mask tensor with rank 2, got {tensor.ndim}")
    return tensor
