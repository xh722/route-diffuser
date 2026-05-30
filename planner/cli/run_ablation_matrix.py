"""Run a small ablation matrix and summarize the results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from planner.cli.common import load_dataset_bundle, load_planner_model, resolve_device
from planner.common import load_yaml_config, set_seed
from planner.trainers import evaluate_model_detailed

DEFAULT_SCORER_CONFIGS = [
    "configs/model/heuristic_only.yaml",
    "configs/model/learned_scorer_light.yaml",
    "configs/model/learned_scorer.yaml",
    "configs/model/learned_scorer_strong.yaml",
    "configs/model/learned_scorer_drift.yaml",
    "configs/model/learned_scorer_mixed.yaml",
    "configs/model/learned_scorer_route_anchor.yaml",
    "configs/model/learned_scorer_reward.yaml",
]
DEFAULT_ENCODER_CONFIGS = [
    "configs/model/base.yaml",
    "configs/model/encoder_small.yaml",
    "configs/model/encoder_wide.yaml",
]
DEFAULT_PRESETS = {
    "scorer": DEFAULT_SCORER_CONFIGS,
    "encoder": DEFAULT_ENCODER_CONFIGS,
}
DEFAULT_BASELINES = {
    "scorer": "configs/model/heuristic_only.yaml",
    "encoder": "configs/model/base.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--inference-config", default="configs/inference/base.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output-dir", default="outputs/ablations/scorer_matrix")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--matrix",
        choices=["scorer", "encoder"],
        default="scorer",
    )
    parser.add_argument(
        "--model-configs",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--baseline-config",
        default="",
    )
    return parser.parse_args()


def _metric_deltas(
    baseline: dict[str, float],
    current: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(baseline) & set(current))
    return {key: round(current[key] - baseline[key], 6) for key in keys}


def _config_name(path: str) -> str:
    return Path(path).stem


def _resolve_model_configs(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.model_configs:
        model_configs = list(args.model_configs)
    else:
        model_configs = list(DEFAULT_PRESETS[args.matrix])
    baseline_config = args.baseline_config or DEFAULT_BASELINES[args.matrix]
    return model_configs, baseline_config


def _summarize_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "hidden_dim": int(model_config.get("hidden_dim", 0)),
        "time_dim": int(model_config.get("time_dim", 0)),
        "decoder_down_dims": list(model_config.get("decoder_down_dims", [])),
        "learned_scorer_weight": float(model_config.get("learned_scorer_weight", 0.0)),
        "scorer_candidate_strategy": str(
            model_config.get("scorer_candidate_strategy", "gt_prior_noise")
        ),
        "scorer_target_mode": str(model_config.get("scorer_target_mode", "ade")),
        "scene_fusion_mode": str(model_config.get("scene_fusion_mode", "concat_mlp")),
    }


def build_ablation_matrix_payload(
    runs: list[dict[str, Any]],
    *,
    matrix_name: str,
    baseline_config_name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not runs:
        raise ValueError("Ablation matrix requires at least one run")

    baseline_run = next(
        (run for run in runs if run["config_name"] == baseline_config_name),
        None,
    )
    if baseline_run is None:
        raise ValueError(
            f"Baseline config {baseline_config_name!r} is not present in ablation runs"
        )

    comparison_rows = []
    for run in runs:
        comparison_rows.append(
            {
                "config_name": run["config_name"],
                "model_config": run["model_config"],
                "config_summary": run["config_summary"],
                "selection_strategy": run["selection_strategy"],
                "overall_metrics": run["overall_metrics"],
                "candidate_set_metrics": run["candidate_set_metrics"],
                "delta_overall_vs_baseline": _metric_deltas(
                    baseline_run["overall_metrics"],
                    run["overall_metrics"],
                ),
                "delta_candidate_vs_baseline": _metric_deltas(
                    baseline_run["candidate_set_metrics"],
                    run["candidate_set_metrics"],
                ),
            }
        )

    return {
        "report_type": f"{matrix_name}_ablation_matrix",
        "matrix_name": matrix_name,
        "baseline_config_name": baseline_config_name,
        "runs": comparison_rows,
        "metadata": metadata,
    }


def build_ablation_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# RouteDiffuser {payload['matrix_name'].title()} Ablation Matrix",
        "",
        f"- Baseline: `{payload['baseline_config_name']}`",
        "",
        "| Config | Hidden | Weight | Candidate Strategy | Target Mode | Fusion | Selection | ADE | FDE | Route Error | Box Collision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in payload["runs"]:
        metrics = run["overall_metrics"]
        summary = run["config_summary"]
        lines.append(
            "| "
            f"`{run['config_name']}` | "
            f"{summary['hidden_dim']} | "
            f"{summary['learned_scorer_weight']:.2f} | "
            f"`{summary['scorer_candidate_strategy']}` | "
            f"`{summary['scorer_target_mode']}` | "
            f"`{summary['scene_fusion_mode']}` | "
            f"`{run['selection_strategy']}` | "
            f"{metrics.get('ade', float('nan')):.3f} | "
            f"{metrics.get('fde', float('nan')):.3f} | "
            f"{metrics.get('route_error', float('nan')):.3f} | "
            f"{metrics.get('box_collision_rate', metrics.get('collision_rate', float('nan'))):.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    infer_config = load_yaml_config(args.inference_config)
    seed = int(infer_config.get("seed", 7))

    batch_size = args.batch_size or int(infer_config.get("batch_size", 4))
    model_config_paths, baseline_config = _resolve_model_configs(args)
    data_config, _, dataset_config, dataloader = load_dataset_bundle(
        args.data_config,
        batch_size=batch_size,
        shuffle=False,
        manifest_path=args.manifest_path,
    )
    checkpoint_path = args.checkpoint or str(infer_config.get("checkpoint_path", ""))
    device = resolve_device(args.device or "cpu")

    num_samples = args.num_samples or int(infer_config.get("num_samples", 3))
    run_summaries: list[dict[str, Any]] = []
    for model_config_path in model_config_paths:
        set_seed(seed)
        model_config, model = load_planner_model(
            model_config_path,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        report = evaluate_model_detailed(
            model=model,
            dataloader=dataloader,
            device=device,
            num_samples=num_samples,
            time_delta=dataset_config.time_delta,
            dataset_name=str(getattr(dataset_config, "dataset_name", "unknown_dataset")),
            dataset_type=str(data_config.get("dataset_type", "synthetic")),
            split="eval",
            metadata={
                "checkpoint_path": checkpoint_path,
                "device": str(device),
                "batch_size": batch_size,
            },
            selection_mode="auto",
        )
        run_summaries.append(
            {
                "config_name": _config_name(model_config_path),
                "model_config": model_config_path,
                "config_summary": _summarize_model_config(model_config),
                "selection_strategy": report.selection.strategy,
                "overall_metrics": report.overall_metrics,
                "candidate_set_metrics": report.candidate_set_metrics,
            }
        )

    baseline_name = _config_name(baseline_config)
    payload = build_ablation_matrix_payload(
        run_summaries,
        matrix_name=args.matrix,
        baseline_config_name=baseline_name,
        metadata={
            "dataset_name": str(getattr(dataset_config, "dataset_name", "unknown_dataset")),
            "dataset_type": str(data_config.get("dataset_type", "synthetic")),
            "checkpoint_path": checkpoint_path,
            "device": str(device),
            "batch_size": batch_size,
            "num_samples": num_samples,
            "model_configs": model_config_paths,
        },
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.matrix}_ablation_matrix.json"
    markdown_path = output_dir / f"{args.matrix}_ablation_matrix.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(build_ablation_markdown(payload), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
