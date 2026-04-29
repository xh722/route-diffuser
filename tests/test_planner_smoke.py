from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from planner.datasets import (
    SyntheticDatasetConfig,
    SyntheticPlanningDataset,
    collate_scene_batches,
)
from planner.models import DiffusionPlanner, DiffusionPlannerConfig
from planner.trainers import create_optimizer, evaluate_model_detailed, train_one_epoch


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


def test_learned_scorer_emits_candidate_logits() -> None:
    batch = build_batch()
    model = DiffusionPlanner(DiffusionPlannerConfig())
    predictions = model.sample(batch, num_samples=3)
    scores = model.score_trajectories(batch, predictions)

    assert scores.shape == (batch.batch_size, 3)


def test_training_loss_returns_finite_scorer_terms_when_enabled() -> None:
    batch = build_batch()
    model = DiffusionPlanner(
        DiffusionPlannerConfig(
            diffusion_steps=4,
            learned_scorer_weight=0.1,
            scorer_num_candidates=3,
        )
    )
    outputs = model.training_loss(batch)

    assert torch.isfinite(outputs["loss"])
    assert torch.isfinite(outputs["diffusion_loss"])
    assert torch.isfinite(outputs["scorer_loss"])
    assert 0.0 <= float(outputs["scorer_accuracy"].item()) <= 1.0


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


def test_detailed_evaluation_reports_scenarios() -> None:
    dataset_config = SyntheticDatasetConfig(num_samples=4)
    dataset = SyntheticPlanningDataset(dataset_config)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_scene_batches,
    )
    model = DiffusionPlanner(DiffusionPlannerConfig(diffusion_steps=4))

    report = evaluate_model_detailed(
        model=model,
        dataloader=dataloader,
        device="cpu",
        num_samples=2,
        time_delta=dataset_config.time_delta,
        dataset_name=dataset_config.dataset_name,
        dataset_type="synthetic",
    )

    assert report.selection.num_samples == 2
    assert report.dataset.name == dataset_config.dataset_name
    assert report.dataset.dataset_type == "synthetic"
    assert "ade" in report.overall_metrics
    assert "oracle_ade" in report.candidate_set_metrics
    assert set(report.scenario_metrics) == {
        "keep_lane",
        "lane_change_left",
        "lane_change_right",
        "gentle_curve",
    }
