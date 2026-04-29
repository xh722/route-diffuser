from __future__ import annotations

from planner.cli.run_ablation_matrix import build_ablation_matrix_payload


def test_ablation_matrix_payload_builds_deltas() -> None:
    payload = build_ablation_matrix_payload(
        [
            {
                "config_name": "heuristic_only",
                "model_config": "configs/model/heuristic_only.yaml",
                "learned_scorer_weight": 0.0,
                "selection_strategy": "heuristic_route_clearance_comfort_scoring",
                "overall_metrics": {"ade": 1.0, "fde": 2.0},
                "candidate_set_metrics": {"oracle_ade": 0.8},
            },
            {
                "config_name": "learned_scorer",
                "model_config": "configs/model/learned_scorer.yaml",
                "learned_scorer_weight": 0.1,
                "selection_strategy": "hybrid_route_clearance_comfort_learned",
                "overall_metrics": {"ade": 0.9, "fde": 1.8},
                "candidate_set_metrics": {"oracle_ade": 0.75},
            },
        ],
        baseline_config_name="heuristic_only",
        metadata={"dataset_name": "route_diffuser_synthetic"},
    )

    assert payload["report_type"] == "scorer_ablation_matrix"
    assert payload["baseline_config_name"] == "heuristic_only"
    assert payload["runs"][1]["delta_overall_vs_baseline"]["ade"] == -0.1
    assert payload["runs"][1]["delta_candidate_vs_baseline"]["oracle_ade"] == -0.05
