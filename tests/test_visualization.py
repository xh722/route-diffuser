from __future__ import annotations

import torch

from planner.datasets import (
    SyntheticDatasetConfig,
    SyntheticPlanningDataset,
    collate_scene_batches,
)
from planner.visualization import (
    plot_candidate_trajectories,
    plot_scenario_gallery,
    plot_trajectory_comparison,
)


def test_plot_trajectory_comparison_handles_batched_route_inputs(tmp_path) -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=2))
    batch = collate_scene_batches([dataset[0], dataset[1]])
    output_path = tmp_path / "trajectory_plot.png"

    result = plot_trajectory_comparison(
        predicted=batch.future_ego_trajectory,
        target=batch.future_ego_trajectory,
        route_polylines=batch.route_lanes,
        route_mask=batch.route_lanes_mask,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_candidate_and_gallery_plots_are_generated(tmp_path) -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=4))
    batch = collate_scene_batches([dataset[index] for index in range(4)])
    predictions = batch.future_ego_trajectory.unsqueeze(1).repeat(1, 3, 1, 1)

    candidate_path = tmp_path / "candidate_plot.png"
    gallery_path = tmp_path / "gallery_plot.png"

    candidate_result = plot_candidate_trajectories(
        predicted_samples=predictions,
        target=batch.future_ego_trajectory,
        route_polylines=batch.route_lanes,
        route_mask=batch.route_lanes_mask,
        output_path=candidate_path,
    )
    gallery_result = plot_scenario_gallery(
        predicted_samples=predictions,
        target=batch.future_ego_trajectory,
        route_polylines=batch.route_lanes,
        route_mask=batch.route_lanes_mask,
        scenario_names=list(batch.metadata["scenario_names"]),
        output_path=gallery_path,
    )

    assert candidate_result == candidate_path
    assert gallery_result == gallery_path
    assert candidate_path.exists()
    assert gallery_path.exists()
