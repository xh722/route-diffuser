from __future__ import annotations

from pathlib import Path

from planner.datasets import (
    DatasetStatistics,
    SyntheticDatasetConfig,
    SyntheticPlanningDataset,
    compute_dataset_statistics,
)


def test_compute_dataset_statistics_and_roundtrip(tmp_path: Path) -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=4))
    stats = compute_dataset_statistics(
        dataset,
        dataset_name=dataset.config.dataset_name,
        dataset_type="synthetic",
        batch_size=2,
        max_scenes=3,
    )
    output_path = tmp_path / "dataset_stats.json"
    stats.save(output_path)
    loaded = DatasetStatistics.load(output_path)

    assert stats.scene_count == 3
    assert "ego_current_state" in stats.tensors
    assert "neighbor_history" in stats.tensors
    assert loaded.dataset_name == dataset.config.dataset_name
    assert loaded.tensors["ego_current_state"].count == 3
    assert len(loaded.tensors["ego_current_state"].mean) == 6
