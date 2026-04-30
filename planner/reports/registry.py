"""Experiment registry and leaderboard helpers for RouteDiffuser artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


@dataclass
class RegistryEntry:
    """One normalized experiment record in the registry."""

    experiment_type: str
    name: str
    source_path: str
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_type": self.experiment_type,
            "name": self.name,
            "source_path": self.source_path,
            "metrics": {key: _round_float(value) for key, value in sorted(self.metrics.items())},
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass
class ExperimentRegistry:
    """Structured registry of experiment outputs."""

    entries: list[RegistryEntry]
    report_type: str = "experiment_registry"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_type": self.report_type,
            "metadata": dict(sorted(self.metadata.items())),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping at {path}")
    return payload


def _evaluation_entry(path: Path, payload: dict[str, Any]) -> RegistryEntry:
    dataset = dict(payload.get("dataset", {}))
    selection = dict(payload.get("selection", {}))
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "dataset_name": dataset.get("name", "unknown"),
            "dataset_type": dataset.get("dataset_type", "unknown"),
            "selection_strategy": selection.get("strategy", "unknown"),
            "report_schema": payload.get("schema_version", "unknown"),
        }
    )
    return RegistryEntry(
        experiment_type="evaluation",
        name=path.stem,
        source_path=str(path),
        metrics={key: float(value) for key, value in dict(payload.get("overall_metrics", {})).items()},
        metadata=metadata,
    )


def _failure_entry(path: Path, payload: dict[str, Any]) -> RegistryEntry:
    overall_failures = list(payload.get("overall_top_failures", []))
    top_case = overall_failures[0] if overall_failures else {}
    metrics = {
        "top_ranking_metric": float(top_case.get("ranking_metric", float("nan"))),
        "top_case_fde": float(dict(top_case.get("metrics", {})).get("fde", float("nan"))),
        "top_case_ade": float(dict(top_case.get("metrics", {})).get("ade", float("nan"))),
    }
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "ranking_metric": payload.get("ranking_metric", "unknown"),
            "selection_strategy": dict(payload.get("selection", {})).get("strategy", "unknown"),
            "top_k": int(payload.get("top_k", 0)),
        }
    )
    return RegistryEntry(
        experiment_type="failure_analysis",
        name=path.stem,
        source_path=str(path),
        metrics=metrics,
        metadata=metadata,
    )


def _ablation_entry(path: Path, payload: dict[str, Any]) -> RegistryEntry:
    runs = list(payload.get("runs", []))
    metrics: dict[str, float] = {"num_runs": float(len(runs))}
    if runs:
        baseline_run = runs[0]
        metrics["baseline_ade"] = float(dict(baseline_run.get("overall_metrics", {})).get("ade", float("nan")))
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "matrix_name": payload.get("matrix_name", "unknown"),
            "baseline_config_name": payload.get("baseline_config_name", "unknown"),
        }
    )
    return RegistryEntry(
        experiment_type="ablation_matrix",
        name=path.stem,
        source_path=str(path),
        metrics=metrics,
        metadata=metadata,
    )


def registry_entry_from_file(path: str | Path) -> RegistryEntry:
    """Load one artifact file into a normalized registry entry."""

    artifact_path = Path(path)
    payload = _load_json(artifact_path)
    report_type = str(payload.get("report_type", ""))

    if report_type == "open_loop_evaluation":
        return _evaluation_entry(artifact_path, payload)
    if report_type == "failure_analysis":
        return _failure_entry(artifact_path, payload)
    if report_type in {"scorer_ablation_matrix", "encoder_ablation_matrix"}:
        return _ablation_entry(artifact_path, payload)

    raise ValueError(f"Unsupported registry artifact report_type {report_type!r} for {artifact_path}")


def build_experiment_registry(paths: list[str | Path], *, metadata: dict[str, Any] | None = None) -> ExperimentRegistry:
    """Build a registry from a list of report artifact paths."""

    entries = [registry_entry_from_file(path) for path in paths]
    entries.sort(key=lambda entry: (entry.experiment_type, entry.name))
    return ExperimentRegistry(entries=entries, metadata={} if metadata is None else metadata)


def build_leaderboard_markdown(registry: ExperimentRegistry, *, primary_metric: str = "fde") -> str:
    """Render a compact leaderboard for evaluation entries."""

    evaluation_entries = [entry for entry in registry.entries if entry.experiment_type == "evaluation"]
    evaluation_entries.sort(key=lambda entry: entry.metrics.get(primary_metric, float("inf")))
    failure_entries = {
        entry.metadata.get("dataset_name", ""): entry
        for entry in registry.entries
        if entry.experiment_type == "failure_analysis"
    }

    lines = [
        "# RouteDiffuser Experiment Leaderboard",
        "",
        f"- Primary Metric: `{primary_metric}`",
        "",
        "| Name | Dataset | Selection | ADE | FDE | Route Error | Collision Rate | Worst-Case FDE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in evaluation_entries:
        metrics = entry.metrics
        failure_entry = failure_entries.get(entry.metadata.get("dataset_name", ""))
        worst_case_fde = (
            failure_entry.metrics.get("top_case_fde", float("nan"))
            if failure_entry is not None
            else float("nan")
        )
        lines.append(
            "| "
            f"`{entry.name}` | "
            f"`{entry.metadata.get('dataset_name', 'unknown')}` | "
            f"`{entry.metadata.get('selection_strategy', 'unknown')}` | "
            f"{metrics.get('ade', float('nan')):.3f} | "
            f"{metrics.get('fde', float('nan')):.3f} | "
            f"{metrics.get('route_error', float('nan')):.3f} | "
            f"{metrics.get('collision_rate', float('nan')):.3f} | "
            f"{worst_case_fde:.3f} |"
        )

    return "\n".join(lines) + "\n"
