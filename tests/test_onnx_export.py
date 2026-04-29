from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from planner.datasets import SyntheticDatasetConfig, SyntheticPlanningDataset, collate_scene_batches
from planner.export import (
    OnnxExportConfig,
    PlannerCoreOnnxWrapper,
    build_export_metadata,
    build_onnx_example_inputs,
)
from planner.models import DiffusionPlanner, DiffusionPlannerConfig


def test_onnx_wrapper_matches_core_output_shape() -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=2))
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_scene_batches,
    )
    batch = next(iter(dataloader))
    model = DiffusionPlanner(DiffusionPlannerConfig(diffusion_steps=4))
    wrapper = PlannerCoreOnnxWrapper(model).eval()
    example_inputs = build_onnx_example_inputs(batch, model)

    output = wrapper(*example_inputs)

    assert output.shape == (
        batch.batch_size,
        model.config.future_horizon,
        model.config.trajectory_dim,
    )


def test_onnx_export_metadata_contains_shapes() -> None:
    dataset = SyntheticPlanningDataset(SyntheticDatasetConfig(num_samples=1))
    batch = collate_scene_batches([dataset[0]])
    model = DiffusionPlanner(DiffusionPlannerConfig(diffusion_steps=4))

    metadata = build_export_metadata(
        model,
        batch,
        export_config=OnnxExportConfig(opset_version=17, dynamic_batch=True),
    )

    assert metadata["export_type"] == "planner_denoiser_core"
    assert metadata["input_shapes"]["ego_current_state"] == [1, 6]
    assert metadata["output_shapes"]["predicted_noise"] == [1, 16, 6]
