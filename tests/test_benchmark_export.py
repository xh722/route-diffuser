from __future__ import annotations

from torch.utils.data import DataLoader

from planner.datasets import SyntheticDatasetConfig, SyntheticPlanningDataset, collate_scene_batches
from planner.export import BenchmarkConfig, benchmark_torch_denoiser_core
from planner.models import DiffusionPlanner, DiffusionPlannerConfig


def test_benchmark_report_contains_latency_fields() -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=2))
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_scene_batches,
    )
    batch = next(iter(dataloader))
    model = DiffusionPlanner(DiffusionPlannerConfig(diffusion_steps=4))

    report = benchmark_torch_denoiser_core(
        model,
        batch,
        BenchmarkConfig(warmup_iterations=0, measure_iterations=1),
    )

    assert report["report_type"] == "torch_denoiser_core_benchmark"
    assert report["batch_size"] == 2
    assert report["output_shape"] == [2, 16, 6]
    assert report["mean_latency_ms"] >= 0.0
    assert report["throughput_samples_per_sec"] >= 0.0
