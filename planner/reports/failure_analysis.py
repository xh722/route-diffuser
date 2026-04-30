"""Scenario-specific failure analysis report schema and collection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch

from planner.datasets.schema import CanonicalSceneBatch
from planner.inference import score_trajectory_candidates_for_mode
from planner.metrics.trajectory import candidate_set_metrics, compute_open_loop_metrics
from planner.models import DiffusionPlanner
from planner.reports.evaluation import EvaluationDatasetInfo, EvaluationSelection

SCHEMA_VERSION = "route_diffuser.failure_analysis.v1"


def _round_metric_map(metrics: dict[str, float], digits: int = 6) -> dict[str, float]:
    return {key: round(float(metrics[key]), digits) for key in sorted(metrics)}


@dataclass
class FailureCase:
    """One per-scene failure record."""

    scene_id: str
    scene_index: int
    scenario_name: str
    ranking_metric: float
    metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    selection: dict[str, float | int | str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_index": self.scene_index,
            "scenario_name": self.scenario_name,
            "ranking_metric": self.ranking_metric,
            "metrics": _round_metric_map(self.metrics),
            "candidate_metrics": _round_metric_map(self.candidate_metrics),
            "selection": self.selection,
        }


@dataclass
class FailureAnalysisReport:
    """Structured report of worst-case scenes under a chosen ranking metric."""

    dataset: EvaluationDatasetInfo
    selection: EvaluationSelection
    ranking_metric: str
    top_k: int
    overall_top_failures: list[FailureCase]
    scenario_top_failures: dict[str, list[FailureCase]]
    project_name: str = "RouteDiffuser"
    report_type: str = "failure_analysis"
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "report_type": self.report_type,
            "dataset": self.dataset.to_dict(),
            "selection": self.selection.to_dict(),
            "ranking_metric": self.ranking_metric,
            "top_k": self.top_k,
            "overall_top_failures": [case.to_dict() for case in self.overall_top_failures],
            "scenario_top_failures": {
                key: [case.to_dict() for case in value]
                for key, value in sorted(self.scenario_top_failures.items())
            },
            "artifacts": {key: str(value) for key, value in sorted(self.artifacts.items())},
            "metadata": dict(sorted(self.metadata.items())),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination

    def save_markdown(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_markdown(), encoding="utf-8")
        return destination

    def to_markdown(self) -> str:
        lines = [
            f"# {self.project_name} Failure Analysis",
            "",
            f"- Ranking Metric: `{self.ranking_metric}`",
            f"- Top K: {self.top_k}",
            f"- Dataset: `{self.dataset.name}`",
            f"- Dataset Type: `{self.dataset.dataset_type}`",
            f"- Selection Strategy: `{self.selection.strategy}`",
            "",
            "## Overall Top Failures",
        ]
        for case in self.overall_top_failures:
            metrics = ", ".join(f"{key}={value:.3f}" for key, value in case.metrics.items())
            lines.append(
                f"- `{case.scene_id}` ({case.scenario_name}): {self.ranking_metric}={case.ranking_metric:.3f}, {metrics}"
            )

        lines.extend(["", "## Scenario Top Failures"])
        for scenario_name, cases in sorted(self.scenario_top_failures.items()):
            lines.append(f"- `{scenario_name}`:")
            for case in cases:
                metrics = ", ".join(f"{key}={value:.3f}" for key, value in case.metrics.items())
                lines.append(
                    f"  - `{case.scene_id}`: {self.ranking_metric}={case.ranking_metric:.3f}, {metrics}"
                )

        return "\n".join(lines) + "\n"


def _selection_strategy_name(selection_mode: str, model: DiffusionPlanner) -> str:
    if selection_mode == "hybrid" or (
        selection_mode == "auto"
        and float(getattr(model.config, "learned_scorer_weight", 0.0)) > 0.0
    ):
        return "hybrid_route_clearance_comfort_learned"
    return "heuristic_route_clearance_comfort_scoring"


def _scene_case(
    batch: CanonicalSceneBatch,
    batch_index: int,
    metrics: dict[str, torch.Tensor],
    candidate_metrics: dict[str, torch.Tensor],
    scored: dict[str, torch.Tensor],
    ranking_metric: str,
) -> FailureCase:
    indices = list(batch.metadata.get("indices", []))
    scenario_names = list(batch.metadata.get("scenario_names", []))
    entry_ids = list(batch.metadata.get("entry_ids", indices))
    scene_index = indices[batch_index] if batch_index < len(indices) else batch_index
    scene_id = str(entry_ids[batch_index]) if batch_index < len(entry_ids) else str(scene_index)
    scenario_name = (
        str(scenario_names[batch_index]) if batch_index < len(scenario_names) else "unknown"
    )

    metric_values = {key: float(value[batch_index].item()) for key, value in metrics.items()}
    candidate_values = {
        key: float(value[batch_index].item()) for key, value in candidate_metrics.items()
    }
    ranking_value = metric_values.get(ranking_metric)
    if ranking_value is None:
        raise ValueError(f"Unknown ranking_metric {ranking_metric!r}")

    selected_index = int(scored["selected_indices"][batch_index].item())
    selected_score = float(scored["scores"][batch_index, selected_index].item())
    selection = {
        "selected_index": selected_index,
        "selected_score": selected_score,
    }
    if scored.get("learned_scores") is not None:
        selection["selected_learned_score"] = float(
            scored["learned_scores"][batch_index, selected_index].item()
        )

    return FailureCase(
        scene_id=scene_id,
        scene_index=int(scene_index),
        scenario_name=scenario_name,
        ranking_metric=ranking_value,
        metrics=metric_values,
        candidate_metrics=candidate_values,
        selection=selection,
    )


@torch.no_grad()
def analyze_failures(
    model: DiffusionPlanner,
    dataloader,
    *,
    device: torch.device | str,
    num_samples: int,
    time_delta: float,
    ranking_metric: str,
    top_k: int,
    dataset_name: str,
    dataset_type: str,
    split: str = "eval",
    selection_mode: str = "auto",
    artifacts: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> FailureAnalysisReport:
    """Collect per-scene metrics and rank the worst cases."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    model.eval()
    all_cases: list[FailureCase] = []
    for batch in dataloader:
        batch = batch.to(device)
        if batch.future_ego_trajectory is None:
            raise ValueError("future_ego_trajectory is required for failure analysis")

        predicted_samples = model.sample(batch, num_samples=num_samples)
        scored = score_trajectory_candidates_for_mode(
            predicted_samples=predicted_samples,
            scene_batch=batch,
            model=model,
            time_delta=time_delta,
            selection_mode=selection_mode,
        )
        selected = scored["selected_trajectories"]
        metrics = compute_open_loop_metrics(
            predicted=selected,
            target=batch.future_ego_trajectory,
            route_polylines=batch.route_lanes,
            route_mask=batch.route_lanes_mask,
            future_mask=batch.future_ego_mask,
            neighbor_history=batch.neighbor_history,
            neighbor_history_mask=batch.neighbor_history_mask,
            time_delta=time_delta,
        )
        candidate_metrics = candidate_set_metrics(
            predicted_samples=predicted_samples,
            target=batch.future_ego_trajectory,
            route_polylines=batch.route_lanes,
            route_mask=batch.route_lanes_mask,
            future_mask=batch.future_ego_mask,
        )

        for batch_index in range(batch.batch_size):
            all_cases.append(
                _scene_case(
                    batch,
                    batch_index,
                    metrics,
                    candidate_metrics,
                    scored,
                    ranking_metric,
                )
            )

    sorted_cases = sorted(all_cases, key=lambda case: case.ranking_metric, reverse=True)
    overall_top = sorted_cases[:top_k]

    scenario_top: dict[str, list[FailureCase]] = {}
    for case in sorted_cases:
        bucket = scenario_top.setdefault(case.scenario_name, [])
        if len(bucket) < top_k:
            bucket.append(case)

    return FailureAnalysisReport(
        dataset=EvaluationDatasetInfo(
            name=dataset_name,
            dataset_type=dataset_type,
            split=split,
        ),
        selection=EvaluationSelection(
            strategy=_selection_strategy_name(selection_mode, model),
            num_samples=num_samples,
            time_delta=time_delta,
        ),
        ranking_metric=ranking_metric,
        top_k=top_k,
        overall_top_failures=overall_top,
        scenario_top_failures=scenario_top,
        artifacts={} if artifacts is None else artifacts,
        metadata={} if metadata is None else metadata,
    )
