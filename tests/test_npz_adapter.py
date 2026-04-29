from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np

from planner.datasets import build_dataset_from_config, build_manifest_from_config, collate_scene_batches


def write_npz_dataset(path: Path) -> None:
    sample_count = 2
    np.savez(
        path,
        ego_current_state=np.zeros((sample_count, 6), dtype=np.float32),
        neighbor_history=np.zeros((sample_count, 3, 4, 6), dtype=np.float32),
        neighbor_history_mask=np.ones((sample_count, 3, 4), dtype=bool),
        lane_polylines=np.zeros((sample_count, 5, 8, 4), dtype=np.float32),
        lane_polylines_mask=np.ones((sample_count, 5, 8), dtype=bool),
        route_lanes=np.zeros((sample_count, 2, 8, 4), dtype=np.float32),
        route_lanes_mask=np.ones((sample_count, 2, 8), dtype=bool),
        future_ego_trajectory=np.zeros((sample_count, 6, 6), dtype=np.float32),
        future_ego_mask=np.ones((sample_count, 6), dtype=bool),
        scenario_name=np.array(["keep_lane", "lane_change_left"]),
    )


def test_npz_manifest_and_dataset_loading(tmp_path: Path) -> None:
    source_path = tmp_path / "planning_dataset.npz"
    write_npz_dataset(source_path)

    config = {
        "dataset_type": "npz",
        "dataset_name": "route_diffuser_npz",
        "source_path": str(source_path),
        "time_delta": 1.0 / 3.0,
    }
    manifest = build_manifest_from_config(config, split="train")
    manifest_path = tmp_path / "planning_manifest.json"
    manifest.save(manifest_path)

    dataset = build_dataset_from_config(
        {
            "dataset_type": "npz",
            "manifest_path": str(manifest_path),
            "source_path": str(source_path),
        }
    )
    batch = collate_scene_batches([dataset[0], dataset[1]])

    assert manifest.adapter_type == "npz"
    assert manifest.scene_count == 2
    assert len(dataset) == 2
    assert dataset[1]["scenario_name"] == "lane_change_left"
    assert batch.batch_size == 2
    assert batch.future_horizon == 6


def test_export_npz_roundtrip_from_synthetic(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "synthetic_manifest.json"
    npz_path = tmp_path / "exported_dataset.npz"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_dataset.py",
            "--output",
            str(manifest_path),
            "--num-samples",
            "3",
        ],
        check=True,
        cwd=repo_root,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/export_dataset_npz.py",
            "--data-config",
            "configs/data/synthetic.yaml",
            "--manifest-path",
            str(manifest_path),
            "--output",
            str(npz_path),
        ],
        check=True,
        cwd=repo_root,
    )

    dataset = build_dataset_from_config(
        {
            "dataset_type": "npz",
            "dataset_name": "route_diffuser_npz",
            "source_path": str(npz_path),
        }
    )

    assert len(dataset) == 3
    assert dataset[0]["future_ego_trajectory"].shape[0] > 0
