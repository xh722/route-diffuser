from __future__ import annotations

from planner.datasets import (
    SyntheticDatasetConfig,
    SyntheticPlanningDataset,
    collate_scene_batches,
)


def test_synthetic_collate_validates_canonical_schema() -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=2))
    batch = collate_scene_batches([dataset[0], dataset[1]])
    batch.validate()

    assert batch.batch_size == 2
    assert batch.ego_current_state.shape[-1] == 6
    assert batch.future_ego_trajectory is not None
    assert batch.future_horizon == 16
