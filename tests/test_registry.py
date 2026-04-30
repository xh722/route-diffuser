from __future__ import annotations

import json
from pathlib import Path

from planner.reports import build_experiment_registry, build_leaderboard_markdown


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_registry_builds_from_multiple_report_types(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "evaluation_report.json"
    failure_path = tmp_path / "failure_analysis.json"
    ablation_path = tmp_path / "scorer_ablation_matrix.json"

    write_json(
        evaluation_path,
        {
            "report_type": "open_loop_evaluation",
            "schema_version": "route_diffuser.evaluation.v1",
            "dataset": {"name": "route_diffuser_synthetic", "dataset_type": "synthetic", "split": "eval"},
            "selection": {"strategy": "heuristic_route_clearance_comfort_scoring", "num_samples": 3, "time_delta": 0.3333333333},
            "overall_metrics": {"ade": 1.0, "fde": 2.0, "route_error": 0.5, "collision_rate": 0.1},
            "candidate_set_metrics": {"oracle_ade": 0.8},
            "scenario_metrics": {},
            "artifacts": {},
            "metadata": {},
        },
    )
    write_json(
        failure_path,
        {
            "report_type": "failure_analysis",
            "dataset": {"name": "route_diffuser_synthetic", "dataset_type": "synthetic", "split": "eval"},
            "selection": {"strategy": "heuristic_route_clearance_comfort_scoring", "num_samples": 3, "time_delta": 0.3333333333},
            "ranking_metric": "fde",
            "top_k": 1,
            "overall_top_failures": [
                {
                    "scene_id": "scene_1",
                    "scene_index": 1,
                    "scenario_name": "keep_lane",
                    "ranking_metric": 3.0,
                    "metrics": {"fde": 3.0},
                    "candidate_metrics": {"oracle_ade": 1.0},
                    "selection": {"selected_index": 0, "selected_score": 1.0},
                }
            ],
            "scenario_top_failures": {},
            "artifacts": {},
            "metadata": {},
        },
    )
    write_json(
        ablation_path,
        {
            "report_type": "scorer_ablation_matrix",
            "matrix_name": "scorer",
            "baseline_config_name": "heuristic_only",
            "runs": [],
            "metadata": {},
        },
    )

    registry = build_experiment_registry(
        [evaluation_path, failure_path, ablation_path],
        metadata={"primary_metric": "fde"},
    )
    leaderboard = build_leaderboard_markdown(registry, primary_metric="fde")

    assert len(registry.entries) == 3
    assert "RouteDiffuser Experiment Leaderboard" in leaderboard
    assert "`evaluation_report`" in leaderboard
    assert "Worst-Case FDE" in leaderboard
