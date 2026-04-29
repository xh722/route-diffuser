"""Parity checking helpers for exported ONNX planner cores."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from planner.datasets.schema import CanonicalSceneBatch
from planner.export.onnx import PlannerCoreOnnxWrapper, build_onnx_example_inputs
from planner.models import DiffusionPlanner


@dataclass(frozen=True)
class OnnxParityConfig:
    """Tolerance settings for ONNX parity checks."""

    atol: float = 1e-4
    rtol: float = 1e-4


def onnx_input_names() -> list[str]:
    """Return the canonical ONNX input order for the denoiser core."""

    return [
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


def tensor_inputs_to_numpy(
    example_inputs: tuple[torch.Tensor, ...],
) -> dict[str, np.ndarray]:
    """Convert example tensor inputs into ONNX runtime feed tensors."""

    feeds: dict[str, np.ndarray] = {}
    for name, value in zip(onnx_input_names(), example_inputs, strict=True):
        feeds[name] = value.detach().cpu().numpy()
    return feeds


def run_torch_denoiser_core(
    model: DiffusionPlanner,
    scene_batch: CanonicalSceneBatch,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run the torch denoiser core and return output plus ONNX-style inputs."""

    wrapper = PlannerCoreOnnxWrapper(model).eval()
    scene_batch = scene_batch.to("cpu")
    example_inputs = build_onnx_example_inputs(scene_batch, model)
    with torch.no_grad():
        output = wrapper(*example_inputs)
    return output.detach().cpu().numpy(), tensor_inputs_to_numpy(example_inputs)


def _load_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ONNX parity checking requires `onnxruntime`. Install it before running parity checks."
        ) from error
    return ort


def run_onnx_denoiser_core(
    onnx_path: str | Path,
    inputs: dict[str, np.ndarray],
) -> np.ndarray:
    """Run one ONNX forward pass for the planner denoiser core."""

    ort = _load_onnxruntime()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(["predicted_noise"], inputs)
    return np.asarray(outputs[0])


def build_parity_report(
    torch_output: np.ndarray,
    onnx_output: np.ndarray,
    config: OnnxParityConfig,
    *,
    onnx_path: str | Path,
) -> dict[str, Any]:
    """Compare torch and ONNX outputs and return a structured report."""

    torch_output = np.asarray(torch_output, dtype=np.float32)
    onnx_output = np.asarray(onnx_output, dtype=np.float32)
    diff = np.abs(torch_output - onnx_output)
    denom = np.maximum(np.abs(torch_output), 1e-8)
    rel_diff = diff / denom
    passed = np.allclose(torch_output, onnx_output, atol=config.atol, rtol=config.rtol)

    return {
        "report_type": "onnx_parity_check",
        "onnx_path": str(onnx_path),
        "atol": config.atol,
        "rtol": config.rtol,
        "passed": bool(passed),
        "output_shape": list(torch_output.shape),
        "max_abs_error": float(diff.max(initial=0.0)),
        "mean_abs_error": float(diff.mean() if diff.size else 0.0),
        "max_rel_error": float(rel_diff.max(initial=0.0)),
    }


def save_parity_report(
    report: dict[str, Any],
    path: str | Path,
) -> Path:
    """Save a parity report as JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination
