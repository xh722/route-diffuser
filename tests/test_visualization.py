from __future__ import annotations

import torch

from planner.datasets import (
    SyntheticDatasetConfig,
    SyntheticPlanningDataset,
    collate_scene_batches,
)
from planner.visualization import plot_trajectory_comparison


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
