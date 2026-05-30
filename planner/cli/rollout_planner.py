"""Public closed-loop rollout entry point for RouteDiffuser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from planner.cli.common import load_dataset_bundle, load_planner_model, resolve_device
from planner.common import load_yaml_config, set_seed
from planner.rollout import rollout_planner, summarize_rollout
from planner.visualization import plot_rollout_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/base.yaml")
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--inference-config", default="configs/inference/base.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output-dir", default="outputs/rollout")
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selection-mode", default="auto")
    return parser.parse_args()


def build_rollout_markdown(summary: dict[str, float | int | str], artifacts: dict[str, str]) -> str:
    lines = [
        "# RouteDiffuser Rollout Summary",
        "",
        f"- Scenario: `{summary['scenario_name']}`",
        f"- Steps: {summary['num_steps']}",
        f"- Closed-loop ADE: {summary['closed_loop_ade']}",
        f"- Closed-loop FDE: {summary['closed_loop_fde']}",
        f"- Route Error: {summary['route_error']}",
        f"- Collision Rate: {summary['collision_rate']}",
        f"- Box Collision Rate: {summary['box_collision_rate']}",
        f"- Point Collision Rate: {summary['point_collision_rate']}",
        f"- Mean Selected Index: {summary['mean_selected_index']}",
        f"- Mean Selected Score: {summary['mean_selected_score']}",
        "",
        "## Artifacts",
    ]
    lines.extend([f"- `{key}`: `{value}`" for key, value in artifacts.items()])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    infer_config = load_yaml_config(args.inference_config)
    set_seed(int(infer_config.get("seed", 7)))

    _, _, dataset_config, dataloader = load_dataset_bundle(
        args.data_config,
        batch_size=1,
        shuffle=False,
        manifest_path=args.manifest_path,
    )
    scene_batch = next(iter(dataloader))

    checkpoint_path = args.checkpoint or str(infer_config.get("checkpoint_path", ""))
    device = resolve_device(args.device or "cpu")
    _, model = load_planner_model(
        args.model_config,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    scene_batch = scene_batch.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = rollout_planner(
        model,
        scene_batch,
        num_steps=args.num_steps,
        num_samples=args.num_samples or int(infer_config.get("num_samples", 3)),
        time_delta=dataset_config.time_delta,
        selection_mode=args.selection_mode,
    )

    trace_path = output_dir / "rollout_trace.pt"
    summary_path = output_dir / "rollout_summary.json"
    markdown_path = output_dir / "rollout_summary.md"
    plot_path = output_dir / "rollout_plot.png"

    plot_rollout_trace(
        executed=result.executed_world_states,
        reference=result.reference_world_states,
        route_polylines=result.route_world_polylines.unsqueeze(0),
        route_mask=result.metadata["route_world_mask"].unsqueeze(0),
        output_path=plot_path,
        title="RouteDiffuser Closed-Loop Rollout",
    )
    torch.save(
        {
            "executed_world_states": result.executed_world_states.cpu(),
            "reference_world_states": None
            if result.reference_world_states is None
            else result.reference_world_states.cpu(),
            "route_world_polylines": result.route_world_polylines.cpu(),
            "selected_indices": result.selected_indices.cpu(),
            "selected_scores": result.selected_scores.cpu(),
            "collision_flags": result.collision_flags.cpu(),
            "point_collision_flags": result.point_collision_flags.cpu(),
            "metadata": {
                "scenario_name": result.metadata["scenario_name"],
                "route_world_mask": result.metadata["route_world_mask"].cpu(),
            },
        },
        trace_path,
    )

    artifacts = {
        "trace": str(trace_path),
        "plot": str(plot_path),
        "json_summary": str(summary_path),
        "markdown_summary": str(markdown_path),
    }
    summary = summarize_rollout(result)
    summary_path.write_text(json.dumps({**summary, "artifacts": artifacts}, indent=2), encoding="utf-8")
    markdown_path.write_text(build_rollout_markdown(summary, artifacts), encoding="utf-8")

    print(json.dumps({**summary, "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
