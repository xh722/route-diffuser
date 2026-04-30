from __future__ import annotations

from torch.utils.data import DataLoader

from planner.datasets import SyntheticDatasetConfig, SyntheticPlanningDataset, collate_scene_batches
from planner.models import DiffusionPlanner, DiffusionPlannerConfig
from planner.reports import analyze_failures


def test_failure_analysis_returns_ranked_cases() -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=4))
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_scene_batches,
    )
    model = DiffusionPlanner(DiffusionPlannerConfig(diffusion_steps=1))

    report = analyze_failures(
        model=model,
        dataloader=dataloader,
        device="cpu",
        num_samples=1,
        time_delta=dataset.config.time_delta,
        ranking_metric="fde",
        top_k=2,
        dataset_name=dataset.config.dataset_name,
        dataset_type="synthetic",
    )

    assert report.ranking_metric == "fde"
    assert len(report.overall_top_failures) == 2
    assert report.overall_top_failures[0].ranking_metric >= report.overall_top_failures[1].ranking_metric
    assert report.scenario_top_failures
