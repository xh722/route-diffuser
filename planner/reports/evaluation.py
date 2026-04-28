"""Structured evaluation report schema for planning experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "route_diffuser.evaluation.v1"


def _sorted_metric_map(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in sorted(metrics)}


def _sorted_nested_metric_map(
    metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        scenario_name: _sorted_metric_map(values)
        for scenario_name, values in sorted(metrics.items())
    }


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


@dataclass(frozen=True)
class EvaluationDatasetInfo:
    """Dataset metadata carried by a structured evaluation report."""

    name: str
    dataset_type: str
    split: str = "eval"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "EvaluationDatasetInfo":
        return cls(
            name=str(values["name"]),
            dataset_type=str(values["dataset_type"]),
            split=str(values.get("split", "eval")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_type": self.dataset_type,
            "split": self.split,
        }


@dataclass(frozen=True)
class EvaluationSelection:
    """Candidate-selection metadata for planning evaluation."""

    strategy: str
    num_samples: int
    time_delta: float

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "EvaluationSelection":
        return cls(
            strategy=str(values["strategy"]),
            num_samples=int(values["num_samples"]),
            time_delta=float(values["time_delta"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "num_samples": self.num_samples,
            "time_delta": self.time_delta,
        }


@dataclass
class EvaluationReport:
    """Stable, serializable planning evaluation report."""

    dataset: EvaluationDatasetInfo
    selection: EvaluationSelection
    overall_metrics: dict[str, float]
    candidate_set_metrics: dict[str, float]
    scenario_metrics: dict[str, dict[str, float]]
    project_name: str = "RouteDiffuser"
    report_type: str = "open_loop_evaluation"
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.overall_metrics = _sorted_metric_map(self.overall_metrics)
        self.candidate_set_metrics = _sorted_metric_map(self.candidate_set_metrics)
        self.scenario_metrics = _sorted_nested_metric_map(self.scenario_metrics)
        self.artifacts = {key: str(value) for key, value in sorted(self.artifacts.items())}
        self.metadata = dict(sorted(self.metadata.items()))

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "EvaluationReport":
        schema_version = str(values.get("schema_version", SCHEMA_VERSION))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported evaluation schema_version {schema_version!r}"
            )
        return cls(
            schema_version=schema_version,
            project_name=str(values.get("project_name", "RouteDiffuser")),
            report_type=str(values.get("report_type", "open_loop_evaluation")),
            dataset=EvaluationDatasetInfo.from_mapping(values["dataset"]),
            selection=EvaluationSelection.from_mapping(values["selection"]),
            overall_metrics={
                key: float(value)
                for key, value in dict(values.get("overall_metrics", {})).items()
            },
            candidate_set_metrics={
                key: float(value)
                for key, value in dict(values.get("candidate_set_metrics", {})).items()
            },
            scenario_metrics={
                str(name): {
                    key: float(metric_value)
                    for key, metric_value in dict(metric_map).items()
                }
                for name, metric_map in dict(values.get("scenario_metrics", {})).items()
            },
            artifacts={
                str(key): str(value)
                for key, value in dict(values.get("artifacts", {})).items()
            },
            metadata=dict(values.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "EvaluationReport":
        report_path = Path(path)
        with report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected mapping at {report_path}")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "report_type": self.report_type,
            "dataset": self.dataset.to_dict(),
            "selection": self.selection.to_dict(),
            "overall_metrics": self.overall_metrics,
            "candidate_set_metrics": self.candidate_set_metrics,
            "scenario_metrics": self.scenario_metrics,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }

    def all_metrics(self) -> dict[str, float]:
        return {
            **self.overall_metrics,
            **self.candidate_set_metrics,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.project_name} Evaluation Report",
            "",
            f"- Schema: `{self.schema_version}`",
            f"- Report Type: `{self.report_type}`",
            f"- Dataset: `{self.dataset.name}`",
            f"- Dataset Type: `{self.dataset.dataset_type}`",
            f"- Split: `{self.dataset.split}`",
            f"- Selection Strategy: `{self.selection.strategy}`",
            f"- Candidate Samples: {self.selection.num_samples}",
            f"- Time Delta: {self.selection.time_delta}",
            "",
            "## Overall Metrics",
        ]
        lines.extend(
            [f"- `{key}`: {_format_metric(value)}" for key, value in self.overall_metrics.items()]
        )
        lines.extend(["", "## Candidate Set Metrics"])
        lines.extend(
            [
                f"- `{key}`: {_format_metric(value)}"
                for key, value in self.candidate_set_metrics.items()
            ]
        )
        lines.extend(["", "## Scenario Metrics"])
        for scenario_name, metrics in self.scenario_metrics.items():
            metric_text = ", ".join(
                f"{key}={_format_metric(value)}" for key, value in metrics.items()
            )
            lines.append(f"- `{scenario_name}`: {metric_text}")

        if self.artifacts:
            lines.extend(["", "## Artifacts"])
            lines.extend([f"- `{key}`: `{value}`" for key, value in self.artifacts.items()])

        if self.metadata:
            lines.extend(["", "## Metadata"])
            lines.extend([f"- `{key}`: `{value}`" for key, value in self.metadata.items()])

        return "\n".join(lines) + "\n"

    def save_json(self, path: str | Path) -> Path:
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return report_path

    def save_markdown(self, path: str | Path) -> Path:
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.to_markdown(), encoding="utf-8")
        return report_path
