"""Lightweight benchmark helpers for deployment-facing planner cores."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from planner.datasets.schema import CanonicalSceneBatch
from planner.export.onnx import PlannerCoreOnnxWrapper, build_onnx_example_inputs
from planner.models import DiffusionPlanner


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a small denoiser-core latency benchmark."""

    warmup_iterations: int = 3
    measure_iterations: int = 10


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_torch_denoiser_core(
    model: DiffusionPlanner,
    scene_batch: CanonicalSceneBatch,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Benchmark the tensor-only denoiser core on one canonical batch."""

    if config.measure_iterations <= 0:
        raise ValueError("measure_iterations must be positive")
    if config.warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")

    wrapper = PlannerCoreOnnxWrapper(model).eval()
    example_inputs = build_onnx_example_inputs(scene_batch, model)
    device = example_inputs[0].device

    with torch.inference_mode():
        for _ in range(config.warmup_iterations):
            _ = wrapper(*example_inputs)
        _synchronize_if_needed(device)

        latencies_ms: list[float] = []
        output_shape: list[int] | None = None
        for _ in range(config.measure_iterations):
            start = time.perf_counter()
            output = wrapper(*example_inputs)
            _synchronize_if_needed(device)
            duration_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(duration_ms)
            if output_shape is None:
                output_shape = list(output.shape)

    latencies = np.asarray(latencies_ms, dtype=np.float64)
    batch_size = scene_batch.batch_size
    mean_ms = float(latencies.mean())
    throughput = float(batch_size / (mean_ms / 1000.0)) if mean_ms > 0.0 else float("inf")
    return {
        "report_type": "torch_denoiser_core_benchmark",
        "device": str(device),
        "warmup_iterations": config.warmup_iterations,
        "measure_iterations": config.measure_iterations,
        "batch_size": batch_size,
        "output_shape": output_shape or [],
        "mean_latency_ms": mean_ms,
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.quantile(latencies, 0.95)),
        "min_latency_ms": float(latencies.min()),
        "max_latency_ms": float(latencies.max()),
        "throughput_samples_per_sec": throughput,
    }


def save_benchmark_report(
    report: dict[str, Any],
    path: str | Path,
) -> Path:
    """Save a benchmark report as JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination
