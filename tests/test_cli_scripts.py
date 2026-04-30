from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_public_cli_entrypoints_expose_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        "scripts/prepare_dataset.py",
        "scripts/compute_dataset_stats.py",
        "scripts/export_dataset_npz.py",
        "scripts/export_onnx.py",
        "scripts/check_onnx_parity.py",
        "scripts/benchmark_infer.py",
        "scripts/rollout_planner.py",
        "scripts/compare_scorer.py",
        "scripts/run_ablation_matrix.py",
        "scripts/analyze_failures.py",
        "scripts/build_registry.py",
        "scripts/train_planner.py",
        "scripts/infer_planner.py",
        "scripts/eval_planner.py",
        "scripts/demo_planner.py",
    ]

    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            check=True,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout
