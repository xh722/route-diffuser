"""Public portfolio demo entry point for RouteDiffuser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from planner.cli.common import apply_overrides, load_dataset_bundle, load_planner_model, resolve_device
from planner.common import load_yaml_config, set_seed
from planner.inference import score_trajectory_candidates
from planner.trainers import create_optimizer, evaluate_model_detailed, train_one_epoch
from planner.visualization import (
    plot_candidate_trajectories,
    plot_scenario_gallery,
    plot_trajectory_comparison,
)

PROJECT_TITLE = "RouteDiffuser"
PROJECT_SUBTITLE = "Autonomous driving trajectory planning with a route-conditioned diffusion policy"
SCENARIO_DESCRIPTIONS = {
    "keep_lane": "Stable forward planning on a straight lane centerline.",
    "lane_change_left": "Lateral transition into the left lane while preserving forward progress.",
    "lane_change_right": "Lateral transition into the right lane while preserving forward progress.",
    "gentle_curve": "Route tracking on a gradually curving road segment.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/base.yaml")
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--train-config", default="configs/train/base.yaml")
    parser.add_argument("--inference-config", default="configs/inference/base.yaml")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output-dir", default="outputs/portfolio_demo")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 3) for key, value in metrics.items()}


def round_nested_metrics(
    metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {key: round_metrics(value) for key, value in metrics.items()}


def build_portfolio_markdown(summary: dict[str, object]) -> str:
    lines = [
        f"# {summary['project_title']}",
        "",
        f"> {summary['subtitle']}",
        "",
        summary["one_liner"],
        "",
        "## Highlights",
    ]
    lines.extend([f"- {item}" for item in summary["highlights"]])
    lines.extend(["", "## Scenario Coverage"])
    lines.extend(
        [
            f"- `{item['name']}`: {item['description']}"
            for item in summary["scenario_catalog"]
        ]
    )
    lines.extend(
        [
            "",
            "## Open-Loop Metrics",
            f"- ADE: {summary['open_loop_metrics']['ade']}",
            f"- FDE: {summary['open_loop_metrics']['fde']}",
            f"- Route Error: {summary['open_loop_metrics']['route_error']}",
            f"- Progress: {summary['open_loop_metrics']['progress']}",
            f"- Min Clearance: {summary['open_loop_metrics']['min_clearance']}",
            f"- Collision Rate: {summary['open_loop_metrics']['collision_rate']}",
            f"- Comfort Violation Rate: {summary['open_loop_metrics']['comfort_violation_rate']}",
            "",
            "## Candidate Set Metrics",
            f"- Oracle ADE: {summary['candidate_metrics']['oracle_ade']}",
            f"- Oracle FDE: {summary['candidate_metrics']['oracle_fde']}",
            f"- Oracle Route Error: {summary['candidate_metrics']['oracle_route_error']}",
            f"- Final-State Diversity: {summary['candidate_metrics']['candidate_final_diversity']}",
            "",
            "## Scenario Breakdown",
        ]
    )
    lines.extend(
        [
            f"- `{name}`: ADE {metrics['ade']}, FDE {metrics['fde']}, Route Error {metrics['route_error']}"
            for name, metrics in summary["scenario_metrics"].items()
        ]
    )
    lines.extend(
        [
            "",
            "## Selection Strategy",
            f"- Strategy: `{summary['selection']['strategy']}`",
            f"- Candidate Samples: {summary['selection']['num_samples']}",
            f"- Mean Selected Index: {summary['open_loop_metrics']['selected_index']}",
            f"- Mean Selected Score: {summary['open_loop_metrics']['selected_score']}",
            "",
            "## Resume Bullets",
        ]
    )
    lines.extend([f"- {item}" for item in summary["resume_bullets"]])
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Checkpoint: `{summary['artifacts']['checkpoint']}`",
            f"- Predictions: `{summary['artifacts']['predictions']}`",
            f"- Plot: `{summary['artifacts']['plot']}`",
            f"- Candidate Plot: `{summary['artifacts']['candidate_plot']}`",
            f"- Scenario Gallery: `{summary['artifacts']['scenario_gallery']}`",
            f"- Evaluation JSON: `{summary['artifacts']['evaluation_json']}`",
            f"- Evaluation Markdown: `{summary['artifacts']['evaluation_markdown']}`",
            f"- JSON Summary: `{summary['artifacts']['json_summary']}`",
            f"- Markdown Summary: `{summary['artifacts']['markdown_summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    train_config = load_yaml_config(args.train_config)
    infer_config = load_yaml_config(args.inference_config)
    train_config = apply_overrides(
        train_config,
        batch_size=args.train_batch_size,
        device=args.device,
    )
    infer_config = apply_overrides(
        infer_config,
        batch_size=args.eval_batch_size,
        num_samples=args.num_samples,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_config.get("seed", infer_config.get("seed", 7)))
    set_seed(seed)

    data_config, dataset, dataset_config, train_dataloader = load_dataset_bundle(
        args.data_config,
        batch_size=int(train_config.get("batch_size", 8)),
        shuffle=True,
        manifest_path=args.manifest_path,
    )
    eval_dataloader = DataLoader(
        dataset,
        batch_size=int(infer_config.get("batch_size", 4)),
        shuffle=False,
        collate_fn=train_dataloader.collate_fn,
    )

    device = resolve_device(str(train_config.get("device", "auto")))
    model_config, model = load_planner_model(
        args.model_config,
        device=device,
    )
    optimizer = create_optimizer(
        model=model,
        learning_rate=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )

    training_curve: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=float(train_config.get("grad_clip_norm", 1.0)),
        )
        training_curve.append({"epoch": float(epoch), **metrics})

    checkpoint_path = output_dir / "demo_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": model_config}, checkpoint_path)

    preview_batch = next(iter(eval_dataloader)).to(device)
    predictions = model.sample(
        preview_batch, num_samples=int(infer_config.get("num_samples", 3))
    )
    scored_preview = score_trajectory_candidates(
        predicted_samples=predictions,
        scene_batch=preview_batch,
        time_delta=dataset_config.time_delta,
    )
    selected_preview = scored_preview["selected_trajectories"]
    predictions_path = output_dir / "predictions.pt"
    torch.save(predictions.cpu(), predictions_path)

    plot_trajectory_comparison(
        predicted=selected_preview,
        target=preview_batch.future_ego_trajectory,
        route_polylines=preview_batch.route_lanes,
        route_mask=preview_batch.route_lanes_mask,
        output_path=output_dir / "prediction_plot.png",
        title="RouteDiffuser Demo Trajectory Comparison",
    )
    plot_candidate_trajectories(
        predicted_samples=predictions,
        target=preview_batch.future_ego_trajectory,
        route_polylines=preview_batch.route_lanes,
        route_mask=preview_batch.route_lanes_mask,
        output_path=output_dir / "candidate_trajectories.png",
        title="RouteDiffuser Candidate Trajectories",
        selected_index=int(scored_preview["selected_indices"][0].item()),
    )
    plot_scenario_gallery(
        predicted_samples=predictions,
        target=preview_batch.future_ego_trajectory,
        route_polylines=preview_batch.route_lanes,
        route_mask=preview_batch.route_lanes_mask,
        scenario_names=list(preview_batch.metadata.get("scenario_names", [])),
        output_path=output_dir / "scenario_gallery.png",
        title="RouteDiffuser Scenario Gallery",
        selected_indices=scored_preview["selected_indices"].detach().cpu().tolist(),
    )

    evaluation_json_path = output_dir / "evaluation_report.json"
    evaluation_markdown_path = output_dir / "evaluation_report.md"
    evaluation_report = evaluate_model_detailed(
        model=model,
        dataloader=eval_dataloader,
        device=device,
        num_samples=int(infer_config.get("num_samples", 3)),
        time_delta=dataset_config.time_delta,
        dataset_name=str(getattr(dataset_config, "dataset_name", "unknown_dataset")),
        dataset_type=str(data_config.get("dataset_type", "synthetic")),
        split="portfolio_demo",
        artifacts={
            "checkpoint": str(checkpoint_path),
            "predictions": str(predictions_path),
            "plot": str(output_dir / "prediction_plot.png"),
            "candidate_plot": str(output_dir / "candidate_trajectories.png"),
            "scenario_gallery": str(output_dir / "scenario_gallery.png"),
            "json_report": str(evaluation_json_path),
            "markdown_report": str(evaluation_markdown_path),
        },
        metadata={
            "epochs": int(args.epochs),
            "device": str(device),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
    )
    evaluation_report.save_json(evaluation_json_path)
    evaluation_report.save_markdown(evaluation_markdown_path)

    scenario_names = list(preview_batch.metadata.get("scenario_names", []))
    rounded_metrics = round_metrics(evaluation_report.overall_metrics)
    rounded_candidate_metrics = round_metrics(evaluation_report.candidate_set_metrics)
    rounded_scenarios = round_nested_metrics(evaluation_report.scenario_metrics)
    training_summary = {
        "epochs": int(args.epochs),
        "initial_loss": round(training_curve[0]["loss"], 4),
        "final_loss": round(training_curve[-1]["loss"], 4),
        "best_loss": round(min(item["loss"] for item in training_curve), 4),
    }

    summary = {
        "project_title": PROJECT_TITLE,
        "subtitle": PROJECT_SUBTITLE,
        "slug": "route-diffuser",
        "one_liner": (
            "A planner-core autonomous driving project that demonstrates canonical scene "
            "modeling, route-conditioned diffusion inference, and end-to-end debug artifacts "
            "without relying on proprietary logs."
        ),
        "tagline": "Conditional 1D U-Net trajectory denoiser for route-conditioned planning",
        "highlights": [
            "Canonical scene schema for ego, neighbors, lanes, route polylines, and masks.",
            "Route-prior residual diffusion with a conditional 1D U-Net decoder.",
            "Optional multi-resolution pyramid noise inspired by a larger reference diffusion planner.",
            "Heuristic candidate ranking over route adherence, clearance, and comfort instead of defaulting to the first sample.",
            "Structured synthetic driving scenarios spanning keep-lane, lane changes, and curves.",
            "End-to-end scripts for training, inference, scenario-level evaluation, and portfolio artifact generation.",
        ],
        "scenario_catalog": [
            {"name": name, "description": SCENARIO_DESCRIPTIONS.get(name, name)}
            for name in scenario_names
        ],
        "training_curve": training_curve,
        "training_summary": training_summary,
        "open_loop_metrics": rounded_metrics,
        "candidate_metrics": rounded_candidate_metrics,
        "scenario_metrics": rounded_scenarios,
        "selection": evaluation_report.selection.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "stack": ["Python", "PyTorch", "Diffusion Models", "Trajectory Planning"],
        "training_config": {
            "diffusion_noise_mode": model_config.get("diffusion_noise_mode", "unknown"),
            "diffusion_noise_discount": model_config.get("diffusion_noise_discount", 0.0),
            "num_samples": int(infer_config.get("num_samples", 3)),
            "time_delta": dataset_config.time_delta,
        },
        "resume_bullets": [
            "Built a route-conditioned autonomous driving planner around a conditional diffusion policy.",
            "Implemented route-prior residual diffusion with a conditional 1D U-Net decoder and iterative denoising sampler.",
            "Added multi-sample candidate scoring, scenario-level evaluation, and multi-resolution diffusion noise inspired by a larger reference planner stack.",
        ],
        "next_extensions": [
            "Swap the synthetic generator with a dataset adapter for logged driving scenes.",
            "Replace heuristic candidate scoring with a learned value or reward model.",
            "Introduce reward modeling or RL fine-tuning on top of the planner boundary.",
        ],
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "predictions": str(predictions_path),
            "plot": str(output_dir / "prediction_plot.png"),
            "candidate_plot": str(output_dir / "candidate_trajectories.png"),
            "scenario_gallery": str(output_dir / "scenario_gallery.png"),
            "evaluation_json": str(evaluation_json_path),
            "evaluation_markdown": str(evaluation_markdown_path),
            "json_summary": str(output_dir / "portfolio_summary.json"),
            "markdown_summary": str(output_dir / "portfolio_summary.md"),
        },
    }
    summary_path = output_dir / "portfolio_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path = output_dir / "portfolio_summary.md"
    markdown_path.write_text(build_portfolio_markdown(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"saved portfolio demo artifacts to {output_dir}")


if __name__ == "__main__":
    main()
