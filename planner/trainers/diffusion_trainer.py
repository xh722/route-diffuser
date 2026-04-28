"""Training helpers for the diffusion planner."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from planner.datasets.schema import CanonicalSceneBatch
from planner.metrics.trajectory import summarize_open_loop_metrics
from planner.models.diffusion_planner import DiffusionPlanner


def create_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Build the default optimizer for the planner."""

    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_one_epoch(
    model: DiffusionPlanner,
    dataloader: Iterable[CanonicalSceneBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    grad_clip_norm: float,
) -> dict[str, float]:
    """Run one training epoch and return aggregate metrics."""

    model.train()
    loss_values: list[float] = []

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        step_outputs = model.training_loss(batch)
        loss = step_outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        loss_values.append(float(loss.item()))

    mean_loss = sum(loss_values) / max(len(loss_values), 1)
    return {"loss": mean_loss, "num_batches": float(len(loss_values))}


@torch.no_grad()
def evaluate_model(
    model: DiffusionPlanner,
    dataloader: Iterable[CanonicalSceneBatch],
    device: torch.device | str,
    num_samples: int = 1,
) -> dict[str, float]:
    """Evaluate the planner with open-loop metrics."""

    model.eval()
    totals = {"ade": 0.0, "fde": 0.0, "route_error": 0.0}
    count = 0

    for batch in dataloader:
        batch = batch.to(device)
        if batch.future_ego_trajectory is None:
            raise ValueError("future_ego_trajectory is required for evaluation")
        predicted = model.sample(batch, num_samples=num_samples)[:, 0]
        metrics = summarize_open_loop_metrics(
            predicted=predicted,
            target=batch.future_ego_trajectory,
            route_polylines=batch.route_lanes,
            route_mask=batch.route_lanes_mask,
            future_mask=batch.future_ego_mask,
        )
        for key, value in metrics.items():
            totals[key] += value
        count += 1

    if count == 0:
        raise ValueError("evaluation dataloader produced zero batches")
    return {key: value / count for key, value in totals.items()}
