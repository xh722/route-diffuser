from __future__ import annotations

import numpy as np

from planner.export import OnnxParityConfig, build_parity_report


def test_parity_report_passes_for_close_outputs() -> None:
    torch_output = np.zeros((1, 16, 6), dtype=np.float32)
    onnx_output = np.full((1, 16, 6), 1e-5, dtype=np.float32)

    report = build_parity_report(
        torch_output,
        onnx_output,
        OnnxParityConfig(atol=1e-4, rtol=1e-4),
        onnx_path="outputs/onnx/planner_denoiser.onnx",
    )

    assert report["passed"] is True
    assert report["output_shape"] == [1, 16, 6]
    assert report["max_abs_error"] > 0.0


def test_parity_report_fails_for_large_error() -> None:
    torch_output = np.zeros((1, 16, 6), dtype=np.float32)
    onnx_output = np.ones((1, 16, 6), dtype=np.float32)

    report = build_parity_report(
        torch_output,
        onnx_output,
        OnnxParityConfig(atol=1e-4, rtol=1e-4),
        onnx_path="outputs/onnx/planner_denoiser.onnx",
    )

    assert report["passed"] is False
    assert report["max_abs_error"] == 1.0
