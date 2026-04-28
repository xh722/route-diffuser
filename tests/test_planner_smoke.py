from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from planner.datasets import (
    SyntheticDatasetConfig,
    SyntheticPlanningDataset,
    collate_scene_batches,
)
from planner.models import DiffusionPlanner, DiffusionPlannerConfig
from planner.trainers import create_optimizer, train_one_epoch


def build_batch(batch_size: int = 2):
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=batch_size))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_scene_batches,
    )
    return next(iter(dataloader))


def test_training_loss_is_finite() -> None:
    batch = build_batch()
    model = DiffusionPlanner(DiffusionPlannerConfig())
    outputs = model.training_loss(batch)

    assert torch.isfinite(outputs["loss"])
    assert outputs["pred_noise"].shape == batch.future_ego_trajectory.shape


def test_sampling_respects_anchor_state() -> None:
    batch = build_batch()
    model = DiffusionPlanner(DiffusionPlannerConfig())
    predictions = model.sample(batch, num_samples=2)

    assert predictions.shape == (batch.batch_size, 2, 16, 6)
    assert torch.allclose(predictions[:, :, 0], batch.ego_current_state.unsqueeze(1))


def test_one_epoch_training_smoke() -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=4))
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_scene_batches,
    )
    model = DiffusionPlanner(DiffusionPlannerConfig())
    optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=1e-4)
    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device="cpu",
        grad_clip_norm=1.0,
    )

    assert metrics["loss"] > 0.0
