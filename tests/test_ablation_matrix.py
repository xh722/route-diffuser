from __future__ import annotations

from planner.cli.run_ablation_matrix import build_ablation_matrix_payload


def test_ablation_matrix_payload_builds_deltas() -> None:
    payload = build_ablation_matrix_payload(
        [
            {
                "config_name": "heuristic_only",
                "model_config": "configs/model/heuristic_only.yaml",
                "config_summary": {"hidden_dim": 128, "learned_scorer_weight": 0.0},
                "selection_strategy": "heuristic_route_clearance_comfort_scoring",
                "overall_metrics": {"ade": 1.0, "fde": 2.0},
                "candidate_set_metrics": {"oracle_ade": 0.8},
            },
            {
                "config_name": "learned_scorer",
                "model_config": "configs/model/learned_scorer.yaml",
                "config_summary": {"hidden_dim": 128, "learned_scorer_weight": 0.1},
                "selection_strategy": "hybrid_route_clearance_comfort_learned",
                "overall_metrics": {"ade": 0.9, "fde": 1.8},
                "candidate_set_metrics": {"oracle_ade": 0.75},
            },
        ],
        matrix_name="scorer",
        baseline_config_name="heuristic_only",
        metadata={"dataset_name": "route_diffuser_synthetic"},
    )

    assert payload["report_type"] == "scorer_ablation_matrix"
    assert payload["baseline_config_name"] == "heuristic_only"
    assert payload["runs"][1]["delta_overall_vs_baseline"]["ade"] == -0.1
    assert payload["runs"][1]["delta_candidate_vs_baseline"]["oracle_ade"] == -0.05


def test_ablation_matrix_payload_supports_encoder_matrix() -> None:
    payload = build_ablation_matrix_payload(
        [
            {
                "config_name": "base",
                "model_config": "configs/model/base.yaml",
                "config_summary": {"hidden_dim": 128, "learned_scorer_weight": 0.0},
                "selection_strategy": "heuristic_route_clearance_comfort_scoring",
                "overall_metrics": {"ade": 1.0},
                "candidate_set_metrics": {"oracle_ade": 0.8},
            },
            {
                "config_name": "encoder_small",
                "model_config": "configs/model/encoder_small.yaml",
                "config_summary": {"hidden_dim": 96, "learned_scorer_weight": 0.0},
                "selection_strategy": "heuristic_route_clearance_comfort_scoring",
                "overall_metrics": {"ade": 1.1},
                "candidate_set_metrics": {"oracle_ade": 0.82},
            },
        ],
        matrix_name="encoder",
        baseline_config_name="base",
        metadata={"dataset_name": "route_diffuser_synthetic"},
    )

    assert payload["report_type"] == "encoder_ablation_matrix"
    assert payload["runs"][1]["config_summary"]["hidden_dim"] == 96
