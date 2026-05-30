"""Training helpers for the diffusion planner."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch
from torch import nn

from planner.datasets.schema import CanonicalSceneBatch
from planner.inference import score_trajectory_candidates_for_mode
from planner.metrics.trajectory import candidate_set_metrics, compute_open_loop_metrics
from planner.models.diffusion_planner import DiffusionPlanner
from planner.reports import EvaluationDatasetInfo, EvaluationReport, EvaluationSelection


def create_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Build the default optimizer for the planner."""

    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_one_epoch(
    model: DiffusionPlanner,
    dataloader: Iterable[CanonicalSceneBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    grad_clip_norm: float,
) -> dict[str, float]:
    """Run one training epoch and return aggregate metrics."""

    model.train()
    metric_sums: dict[str, float] = {
        "loss": 0.0,
        "diffusion_loss": 0.0,
        "scorer_loss": 0.0,
        "scorer_accuracy": 0.0,
    }
    count = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        step_outputs = model.training_loss(batch)
        loss = step_outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        for key in metric_sums:
            metric_sums[key] += float(step_outputs[key].item())
        count += 1

    if count == 0:
        return {key: 0.0 for key in (*metric_sums.keys(), "num_batches")}
    return {
        **{key: value / count for key, value in metric_sums.items()},
        "num_batches": float(count),
    }


def _append_metric_store(
    store: dict[str, list[torch.Tensor]],
    metrics: dict[str, torch.Tensor],
) -> None:
    for key, value in metrics.items():
        store[key].append(value.detach().cpu())


def _reduce_metric_store(store: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for key, values in store.items():
        merged = torch.cat([value.reshape(-1) for value in values], dim=0)
        merged = merged.to(torch.float32)
        finite_mask = torch.isfinite(merged)
        if not finite_mask.any():
            reduced[key] = float("nan")
            continue
        reduced[key] = float(merged[finite_mask].mean().item())
    return reduced


def _normalize_metric_names(metrics: dict[str, float]) -> dict[str, float]:
    normalized = dict(metrics)
    if "comfort_violation" in normalized:
        normalized["comfort_violation_rate"] = normalized.pop("comfort_violation")
    if "collision" in normalized:
        normalized["collision_rate"] = normalized.pop("collision")
    if "box_collision" in normalized:
        normalized["box_collision_rate"] = normalized.pop("box_collision")
    if "point_collision" in normalized:
        normalized["point_collision_rate"] = normalized.pop("point_collision")
    return normalized


@torch.no_grad()
def evaluate_model_detailed(
    model: DiffusionPlanner,
    dataloader: Iterable[CanonicalSceneBatch],
    device: torch.device | str,
    num_samples: int = 1,
    time_delta: float = 1.0 / 3.0,
    project_name: str = "RouteDiffuser",
    dataset_name: str = "unknown_dataset",
    dataset_type: str = "unknown",
    split: str = "eval",
    artifacts: dict[str, str] | None = None,
    metadata: dict[str, object] | None = None,
    selection_mode: str = "auto",
) -> EvaluationReport:
    """Evaluate the planner and return a planning-style report."""

    model.eval()
    overall_store: dict[str, list[torch.Tensor]] = defaultdict(list)
    candidate_store: dict[str, list[torch.Tensor]] = defaultdict(list)
    scenario_store: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    batch_count = 0

    for batch in dataloader:
        batch = batch.to(device)
        if batch.future_ego_trajectory is None:
            raise ValueError("future_ego_trajectory is required for evaluation")

        predicted_samples = model.sample(batch, num_samples=num_samples)
        scored = score_trajectory_candidates_for_mode(
            predicted_samples=predicted_samples,
            scene_batch=batch,
            model=model,
            time_delta=time_delta,
            selection_mode=selection_mode,
        )
        selected = scored["selected_trajectories"]

        open_loop = compute_open_loop_metrics(
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
        open_loop["selected_index"] = scored["selected_indices"].to(torch.float32)
        open_loop["selected_score"] = scored["selected_scores"]
        open_loop["selected_heuristic_score"] = scored["selected_heuristic_scores"]
        open_loop["best_heuristic_score"] = scored["best_heuristic_scores"]
        open_loop["heuristic_regret"] = scored["heuristic_regret"]
        open_loop["matches_heuristic_best"] = (
            scored["selected_indices"] == scored["heuristic_selected_indices"]
        ).to(torch.float32)
        if scored["learned_selected_indices"] is not None:
            open_loop["matches_learned_best"] = (
                scored["selected_indices"] == scored["learned_selected_indices"]
            ).to(torch.float32)
        if scored["selected_normalized_learned_scores"] is not None:
            open_loop["selected_normalized_learned_score"] = scored[
                "selected_normalized_learned_scores"
            ]
        if scored["learned_preference_regret"] is not None:
            open_loop["learned_preference_regret"] = scored["learned_preference_regret"]

        _append_metric_store(overall_store, open_loop)
        _append_metric_store(candidate_store, candidate_metrics)

        scenario_names = list(batch.metadata.get("scenario_names", []))
        if len(scenario_names) != batch.batch_size:
            scenario_names = [f"scenario_{index}" for index in range(batch.batch_size)]

        for sample_index, scenario_name in enumerate(scenario_names):
            for key, value in open_loop.items():
                scenario_store[scenario_name][key].append(
                    value[sample_index : sample_index + 1].detach().cpu()
                )
            for key, value in candidate_metrics.items():
                scenario_store[scenario_name][key].append(
                    value[sample_index : sample_index + 1].detach().cpu()
                )

        batch_count += 1

    if batch_count == 0:
        raise ValueError("evaluation dataloader produced zero batches")

    return EvaluationReport(
        project_name=project_name,
        dataset=EvaluationDatasetInfo(
            name=dataset_name,
            dataset_type=dataset_type,
            split=split,
        ),
        selection=EvaluationSelection(
            strategy=(
                "hybrid_route_clearance_comfort_learned"
                if selection_mode == "hybrid"
                or (
                    selection_mode == "auto"
                    and float(getattr(model.config, "learned_scorer_weight", 0.0)) > 0.0
                )
                else "heuristic_route_clearance_comfort_scoring"
            ),
            num_samples=int(num_samples),
            time_delta=float(time_delta),
        ),
        overall_metrics=_normalize_metric_names(_reduce_metric_store(overall_store)),
        candidate_set_metrics=_reduce_metric_store(candidate_store),
        scenario_metrics={
            scenario_name: _normalize_metric_names(_reduce_metric_store(metrics))
            for scenario_name, metrics in sorted(scenario_store.items())
        },
        artifacts={} if artifacts is None else artifacts,
        metadata={} if metadata is None else metadata,
    )


@torch.no_grad()
def evaluate_model(
    model: DiffusionPlanner,
    dataloader: Iterable[CanonicalSceneBatch],
    device: torch.device | str,
    num_samples: int = 1,
    time_delta: float = 1.0 / 3.0,
    selection_mode: str = "auto",
) -> dict[str, float]:
    """Evaluate the planner with open-loop metrics."""

    report = evaluate_model_detailed(
        model=model,
        dataloader=dataloader,
        device=device,
        num_samples=num_samples,
        time_delta=time_delta,
        selection_mode=selection_mode,
    )
    return report.all_metrics()
