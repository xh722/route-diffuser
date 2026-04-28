from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from planner.datasets import (
    DatasetManifest,
    SyntheticDatasetConfig,
    build_dataset_from_config,
    build_manifest_from_config,
)


def test_manifest_roundtrip_and_dataset_subset(tmp_path: Path) -> None:
    manifest_path = tmp_path / "synthetic_subset.json"
    manifest = build_manifest_from_config(
        {
            "dataset_type": "synthetic",
            "dataset_name": "route_diffuser_synthetic",
            "num_samples": 8,
            "seed": 7,
        },
        split="train",
        start_index=4,
        num_samples=3,
    )
    manifest.save(manifest_path)

    loaded = DatasetManifest.load(manifest_path)
    dataset = build_dataset_from_config(
        {
            "dataset_type": "synthetic",
            "manifest_path": str(manifest_path),
            "seed": 7,
        }
    )

    assert loaded.scene_count == 3
    assert len(loaded.entries) == 3
    assert len(dataset) == 3
    assert dataset[0]["index"] == 4
    assert dataset[1]["scenario_name"] == "lane_change_left"


def test_prepare_dataset_script_writes_manifest(tmp_path: Path) -> None:
    output_path = tmp_path / "prepared_manifest.json"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dataset.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-config",
            "configs/data/synthetic.yaml",
            "--output",
            str(output_path),
            "--split",
            "val",
            "--start-index",
            "2",
            "--num-samples",
            "5",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    manifest = DatasetManifest.load(output_path)
    config = SyntheticDatasetConfig.from_mapping(manifest.config)

    assert manifest.adapter_type == "synthetic"
    assert manifest.split == "val"
    assert manifest.scene_count == 5
    assert config.dataset_name == "route_diffuser_synthetic"
    assert manifest.entries[0].sample_index == 2
