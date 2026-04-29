"""Plotting helpers for predicted and ground-truth trajectories."""

from __future__ import annotations

import math
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

    _plot_route(axis, route_polylines, route_mask, item_index=0)
    axis.plot(target[:, 0], target[:, 1], color="forestgreen", linewidth=2.0, label="target")
    axis.plot(predicted[:, 0], predicted[:, 1], color="firebrick", linewidth=2.0, label="predicted")
    axis.scatter(target[0, 0], target[0, 1], color="black", s=30, label="start")
    _style_axis(axis, title)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
        plt.close(fig)
        return destination

    return fig


def plot_candidate_trajectories(
    predicted_samples: torch.Tensor,
    target: torch.Tensor,
    route_polylines: torch.Tensor | None = None,
    route_mask: torch.Tensor | None = None,
    output_path: str | Path | None = None,
    title: str = "Candidate trajectories",
    selected_index: int | None = None,
):
    """Plot multiple sampled trajectories for the first scene in the batch."""

    candidate_bundle = _select_prediction_bundle(predicted_samples).detach().cpu()
    target = _select_trajectory(target).detach().cpu()
    highlight_index = (
        _best_candidate_index(candidate_bundle, target)
        if selected_index is None
        else int(selected_index)
    )

    fig, axis = plt.subplots(figsize=(6, 6))
    _plot_route(axis, route_polylines, route_mask, item_index=0)
    for candidate_index in range(candidate_bundle.shape[0]):
        color = "firebrick" if candidate_index == highlight_index else "tab:red"
        alpha = 0.95 if candidate_index == highlight_index else 0.35
        linewidth = 2.2 if candidate_index == highlight_index else 1.2
        label = "selected sample" if candidate_index == highlight_index else None
        candidate = candidate_bundle[candidate_index]
        axis.plot(
            candidate[:, 0],
            candidate[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )

    axis.plot(target[:, 0], target[:, 1], color="forestgreen", linewidth=2.0, label="target")
    axis.scatter(target[0, 0], target[0, 1], color="black", s=30, label="start")
    _style_axis(axis, title)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
        plt.close(fig)
        return destination

    return fig


def plot_scenario_gallery(
    predicted_samples: torch.Tensor,
    target: torch.Tensor,
    route_polylines: torch.Tensor,
    route_mask: torch.Tensor,
    scenario_names: list[str] | None = None,
    output_path: str | Path | None = None,
    title: str = "Scenario gallery",
    max_scenarios: int = 4,
    selected_indices: list[int] | None = None,
):
    """Plot multiple scenes in a compact gallery using the first sampled candidate."""

    predicted_samples = predicted_samples.detach().cpu()
    target = target.detach().cpu()
    route_polylines = route_polylines.detach().cpu()
    route_mask = route_mask.detach().cpu()

    if predicted_samples.ndim != 4:
        raise ValueError("predicted_samples must have shape [B, S, T, D]")

    num_scenarios = min(max_scenarios, predicted_samples.shape[0])
    columns = 2
    rows = math.ceil(num_scenarios / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 5 * rows))
    if hasattr(axes, "reshape"):
        axes = axes.reshape(-1)
    else:
        axes = [axes]

    for scenario_index in range(num_scenarios):
        axis = axes[scenario_index]
        _plot_route(axis, route_polylines, route_mask, item_index=scenario_index)
        scenario_target = target[scenario_index]
        scenario_candidates = predicted_samples[scenario_index]
        highlight_index = (
            _best_candidate_index(scenario_candidates, scenario_target)
            if selected_indices is None or scenario_index >= len(selected_indices)
            else int(selected_indices[scenario_index])
        )

        for candidate_index in range(scenario_candidates.shape[0]):
            candidate = scenario_candidates[candidate_index]
            alpha = 0.85 if candidate_index == highlight_index else 0.25
            linewidth = 2.0 if candidate_index == highlight_index else 1.0
            axis.plot(candidate[:, 0], candidate[:, 1], color="tab:red", alpha=alpha, linewidth=linewidth)

        axis.plot(scenario_target[:, 0], scenario_target[:, 1], color="forestgreen", linewidth=2.0)
        axis.scatter(scenario_target[0, 0], scenario_target[0, 1], color="black", s=20)
        scenario_title = (
            scenario_names[scenario_index]
            if scenario_names is not None and scenario_index < len(scenario_names)
            else f"scenario {scenario_index + 1}"
        )
        _style_axis(axis, scenario_title, add_legend=False)

    for axis in axes[num_scenarios:]:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
        plt.close(fig)
        return destination

    return fig


def plot_rollout_trace(
    executed: torch.Tensor,
    reference: torch.Tensor | None = None,
    route_polylines: torch.Tensor | None = None,
    route_mask: torch.Tensor | None = None,
    output_path: str | Path | None = None,
    title: str = "Closed-loop rollout trace",
):
    """Plot a closed-loop rollout against the route and optional reference path."""

    executed = executed.detach().cpu()
    route_polylines = (
        None if route_polylines is None else route_polylines.detach().cpu()
    )
    route_mask = None if route_mask is None else route_mask.detach().cpu()
    reference = None if reference is None else reference.detach().cpu()

    fig, axis = plt.subplots(figsize=(7, 6))
    _plot_route(axis, route_polylines, route_mask, item_index=0)
    if reference is not None:
        axis.plot(
            reference[:, 0],
            reference[:, 1],
            color="forestgreen",
            linewidth=2.0,
            linestyle="--",
            label="reference",
        )
    axis.plot(
        executed[:, 0],
        executed[:, 1],
        color="firebrick",
        linewidth=2.4,
        label="executed",
    )
    axis.scatter(executed[0, 0], executed[0, 1], color="black", s=30, label="start")
    axis.scatter(executed[-1, 0], executed[-1, 1], color="firebrick", s=35, label="final")
    _style_axis(axis, title)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
        plt.close(fig)
        return destination

    return fig


def _best_candidate_index(candidates: torch.Tensor, target: torch.Tensor) -> int:
    errors = torch.linalg.norm(candidates[..., :2] - target.unsqueeze(0)[..., :2], dim=-1)
    return int(errors.mean(dim=-1).argmin().item())


def _plot_route(
    axis: plt.Axes,
    route_polylines: torch.Tensor | None,
    route_mask: torch.Tensor | None,
    item_index: int,
) -> None:
    if route_polylines is None:
        return

    route_polylines = _select_polyline_bank(route_polylines, item_index=item_index).detach().cpu()
    route_mask = (
        None
        if route_mask is None
        else _select_polyline_mask(route_mask, item_index=item_index).detach().cpu()
    )
    for polyline_index in range(route_polylines.shape[0]):
        points = route_polylines[polyline_index]
        if route_mask is not None:
            valid = route_mask[polyline_index].bool()
            points = points[valid]
        if points.numel() == 0:
            continue
        axis.plot(points[:, 0], points[:, 1], color="lightgray", linewidth=1.0)


def _style_axis(axis: plt.Axes, title: str, add_legend: bool = True) -> None:
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.axis("equal")
    if add_legend:
        axis.legend()
    axis.grid(True, linestyle="--", linewidth=0.5)


def _select_trajectory(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor[0, 0]
    elif tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Expected trajectory tensor with rank 2, got {tensor.ndim}")
    return tensor


def _select_prediction_bundle(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor[0]
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected prediction bundle with rank 3, got {tensor.ndim}")
    return tensor


def _select_polyline_bank(tensor: torch.Tensor, item_index: int = 0) -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor[item_index]
    if tensor.ndim != 3:
        raise ValueError(f"Expected polyline bank tensor with rank 3, got {tensor.ndim}")
    return tensor


def _select_polyline_mask(tensor: torch.Tensor, item_index: int = 0) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor[item_index]
    if tensor.ndim != 2:
        raise ValueError(f"Expected polyline mask tensor with rank 2, got {tensor.ndim}")
    return tensor
