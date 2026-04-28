from __future__ import annotations

from pathlib import Path

from planner.reports import EvaluationDatasetInfo, EvaluationReport, EvaluationSelection


def build_report() -> EvaluationReport:
    return EvaluationReport(
        project_name="RouteDiffuser",
        dataset=EvaluationDatasetInfo(
            name="route_diffuser_synthetic",
            dataset_type="synthetic",
            split="eval",
        ),
        selection=EvaluationSelection(
            strategy="heuristic_route_clearance_comfort_scoring",
            num_samples=3,
            time_delta=1.0 / 3.0,
        ),
        overall_metrics={"ade": 1.2, "fde": 1.8},
        candidate_set_metrics={"oracle_ade": 1.0},
        scenario_metrics={"keep_lane": {"ade": 0.9, "fde": 1.1}},
        artifacts={"json_report": "outputs/eval/evaluation_report.json"},
        metadata={"device": "cpu"},
    )


def test_evaluation_report_roundtrip(tmp_path: Path) -> None:
    report = build_report()
    json_path = tmp_path / "evaluation_report.json"
    markdown_path = tmp_path / "evaluation_report.md"

    report.save_json(json_path)
    report.save_markdown(markdown_path)
    loaded = EvaluationReport.load_json(json_path)

    assert loaded.schema_version == "route_diffuser.evaluation.v1"
    assert loaded.dataset.name == "route_diffuser_synthetic"
    assert loaded.selection.num_samples == 3
    assert loaded.overall_metrics["ade"] == 1.2
    assert loaded.scenario_metrics["keep_lane"]["fde"] == 1.1
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# RouteDiffuser Evaluation Report"
    )
