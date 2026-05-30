from __future__ import annotations

import torch

from planner.datasets import (
    SyntheticDatasetConfig,
    SyntheticPlanningDataset,
    collate_scene_batches,
)
from planner.inference import score_trajectory_candidates
from planner.metrics import (
    box_collision_matrix,
    candidate_set_metrics,
    summarize_open_loop_metrics,
)


def test_oriented_box_collision_matrix_detects_overlap() -> None:
    ego = torch.zeros(1, 2, 6)
    ego[..., 2] = 1.0
    objects = torch.zeros(1, 2, 2, 6)
    objects[..., 2] = 1.0
    objects[:, 0, :, 0] = 1.0
    objects[:, 1, :, 0] = 20.0

    collision = box_collision_matrix(ego, objects)

    assert collision.shape == (1, 2, 2)
    assert collision[0, 0].all()
    assert not collision[0, 1].any()


def test_open_loop_metrics_include_behavior_and_clearance() -> None:
    dataset_config = SyntheticDatasetConfig(num_samples=2)
    dataset = SyntheticPlanningDataset(dataset_config)
    batch = collate_scene_batches([dataset[0], dataset[1]])

    metrics = summarize_open_loop_metrics(
        predicted=batch.future_ego_trajectory,
        target=batch.future_ego_trajectory,
        route_polylines=batch.route_lanes,
        route_mask=batch.route_lanes_mask,
        future_mask=batch.future_ego_mask,
        neighbor_history=batch.neighbor_history,
        neighbor_history_mask=batch.neighbor_history_mask,
        time_delta=dataset_config.time_delta,
    )

    assert metrics["ade"] == 0.0
    assert metrics["fde"] == 0.0
    assert metrics["progress"] > 0.0
    assert metrics["min_clearance"] > 0.0
    assert 0.0 <= metrics["collision_rate"] <= 1.0
    assert 0.0 <= metrics["box_collision_rate"] <= 1.0
    assert 0.0 <= metrics["point_collision_rate"] <= 1.0
    assert 0.0 <= metrics["comfort_violation_rate"] <= 1.0


def test_candidate_scoring_prefers_route_consistent_sample() -> None:
    dataset_config = SyntheticDatasetConfig(num_samples=1)
    dataset = SyntheticPlanningDataset(dataset_config)
    batch = collate_scene_batches([dataset[0]])

    good = batch.future_ego_trajectory.clone()
    bad = good.clone()
    bad[:, :, 1] += 8.0
    predictions = torch.stack([good, bad], dim=1)

    scored = score_trajectory_candidates(
        predicted_samples=predictions,
        scene_batch=batch,
        time_delta=dataset_config.time_delta,
    )
    candidate_metrics = candidate_set_metrics(
        predicted_samples=predictions,
        target=batch.future_ego_trajectory,
        route_polylines=batch.route_lanes,
        route_mask=batch.route_lanes_mask,
        future_mask=batch.future_ego_mask,
    )

    assert int(scored["selected_indices"][0].item()) == 0
    assert float(scored["route_error"][0, 0]) < float(scored["route_error"][0, 1])
    assert candidate_metrics["oracle_ade"].shape == (1,)
    assert candidate_metrics["candidate_final_diversity"].shape == (1,)
