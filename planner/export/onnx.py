"""ONNX export helpers for the RouteDiffuser denoiser core."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from planner.datasets.schema import CanonicalSceneBatch
from planner.models import DiffusionPlanner


@dataclass(frozen=True)
class OnnxExportConfig:
    """Configuration for exporting the tensor-only planner core to ONNX."""

    opset_version: int = 17
    dynamic_batch: bool = True


class PlannerCoreOnnxWrapper(nn.Module):
    """Tensor-only wrapper around the planner denoiser core."""

    def __init__(self, model: DiffusionPlanner) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        ego_current_state: torch.Tensor,
        neighbor_history: torch.Tensor,
        neighbor_history_mask: torch.Tensor,
        lane_polylines: torch.Tensor,
        lane_polylines_mask: torch.Tensor,
        route_lanes: torch.Tensor,
        route_lanes_mask: torch.Tensor,
        noisy_trajectory: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        scene_batch = CanonicalSceneBatch(
            ego_current_state=ego_current_state,
            neighbor_history=neighbor_history,
            neighbor_history_mask=neighbor_history_mask,
            lane_polylines=lane_polylines,
            lane_polylines_mask=lane_polylines_mask,
            route_lanes=route_lanes,
            route_lanes_mask=route_lanes_mask,
        ).validate()
        return self.model(scene_batch, noisy_trajectory, timesteps)


def build_onnx_example_inputs(
    scene_batch: CanonicalSceneBatch,
    model: DiffusionPlanner,
) -> tuple[torch.Tensor, ...]:
    """Build example tensor inputs for ONNX export."""

    batch_size = scene_batch.batch_size
    device = scene_batch.ego_current_state.device
    dtype = scene_batch.ego_current_state.dtype

    noisy_trajectory = torch.zeros(
        batch_size,
        model.config.future_horizon,
        model.config.trajectory_dim,
        device=device,
        dtype=dtype,
    )
    timesteps = torch.zeros(batch_size, device=device, dtype=torch.long)
    return (
        scene_batch.ego_current_state,
        scene_batch.neighbor_history,
        scene_batch.neighbor_history_mask,
        scene_batch.lane_polylines,
        scene_batch.lane_polylines_mask,
        scene_batch.route_lanes,
        scene_batch.route_lanes_mask,
        noisy_trajectory,
        timesteps,
    )


def default_dynamic_axes() -> dict[str, dict[int, str]]:
    """Return dynamic axis annotations for exported ONNX tensors."""

    return {
        "ego_current_state": {0: "batch"},
        "neighbor_history": {0: "batch"},
        "neighbor_history_mask": {0: "batch"},
        "lane_polylines": {0: "batch"},
        "lane_polylines_mask": {0: "batch"},
        "route_lanes": {0: "batch"},
        "route_lanes_mask": {0: "batch"},
        "noisy_trajectory": {0: "batch"},
        "timesteps": {0: "batch"},
        "predicted_noise": {0: "batch"},
    }


def export_planner_core_to_onnx(
    model: DiffusionPlanner,
    scene_batch: CanonicalSceneBatch,
    output_path: str | Path,
    *,
    config: OnnxExportConfig | None = None,
) -> Path:
    """Export the planner denoiser core to ONNX."""

    export_config = OnnxExportConfig() if config is None else config
    wrapper = PlannerCoreOnnxWrapper(model).eval()
    example_inputs = build_onnx_example_inputs(scene_batch, model)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_names = [
        "ego_current_state",
        "neighbor_history",
        "neighbor_history_mask",
        "lane_polylines",
        "lane_polylines_mask",
        "route_lanes",
        "route_lanes_mask",
        "noisy_trajectory",
        "timesteps",
    ]
    dynamic_axes = default_dynamic_axes() if export_config.dynamic_batch else None

    try:
        torch.onnx.export(
            wrapper,
            example_inputs,
            destination,
            input_names=input_names,
            output_names=["predicted_noise"],
            opset_version=export_config.opset_version,
            dynamic_axes=dynamic_axes,
        )
    except ModuleNotFoundError as error:
        if error.name == "onnx":
            raise RuntimeError(
                "ONNX export requires the `onnx` package. Install it before exporting."
            ) from error
        raise

    return destination


def build_export_metadata(
    model: DiffusionPlanner,
    scene_batch: CanonicalSceneBatch,
    export_config: OnnxExportConfig,
) -> dict[str, Any]:
    """Describe the exported ONNX interface for downstream tooling."""

    return {
        "export_type": "planner_denoiser_core",
        "opset_version": export_config.opset_version,
        "dynamic_batch": export_config.dynamic_batch,
        "future_horizon": model.config.future_horizon,
        "trajectory_dim": model.config.trajectory_dim,
        "input_shapes": {
            "ego_current_state": list(scene_batch.ego_current_state.shape),
            "neighbor_history": list(scene_batch.neighbor_history.shape),
            "neighbor_history_mask": list(scene_batch.neighbor_history_mask.shape),
            "lane_polylines": list(scene_batch.lane_polylines.shape),
            "lane_polylines_mask": list(scene_batch.lane_polylines_mask.shape),
            "route_lanes": list(scene_batch.route_lanes.shape),
            "route_lanes_mask": list(scene_batch.route_lanes_mask.shape),
            "noisy_trajectory": [
                scene_batch.batch_size,
                model.config.future_horizon,
                model.config.trajectory_dim,
            ],
            "timesteps": [scene_batch.batch_size],
        },
        "output_shapes": {
            "predicted_noise": [
                scene_batch.batch_size,
                model.config.future_horizon,
                model.config.trajectory_dim,
            ]
        },
    }


def save_export_metadata(
    metadata: dict[str, Any],
    path: str | Path,
) -> Path:
    """Save ONNX export metadata alongside the model artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination
