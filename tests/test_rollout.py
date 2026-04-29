from __future__ import annotations

from torch.utils.data import DataLoader

from planner.datasets import SyntheticDatasetConfig, SyntheticPlanningDataset, collate_scene_batches
from planner.models import DiffusionPlanner, DiffusionPlannerConfig
from planner.rollout import rollout_planner, summarize_rollout


def test_rollout_returns_trace_and_summary() -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=1))
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_scene_batches,
    )
    batch = next(iter(dataloader))
    model = DiffusionPlanner(DiffusionPlannerConfig(diffusion_steps=1))

    result = rollout_planner(
        model,
        batch,
        num_steps=2,
        num_samples=1,
        time_delta=1.0 / 3.0,
    )
    summary = summarize_rollout(result)

    assert result.executed_world_states.shape == (3, 6)
    assert result.selected_indices.shape == (2,)
    assert result.selected_scores.shape == (2,)
    assert "closed_loop_ade" in summary
    assert "route_error" in summary
