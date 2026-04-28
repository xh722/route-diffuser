"""Public inference entry point for RouteDiffuser."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from planner.cli.common import apply_overrides, load_dataset_bundle, load_planner_model, resolve_device
from planner.common import load_yaml_config, set_seed
from planner.inference import score_trajectory_candidates
from planner.visualization import plot_trajectory_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/base.yaml")
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--inference-config", default="configs/inference/base.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_config = load_yaml_config(args.inference_config)
    infer_config = apply_overrides(
        infer_config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=args.device,
    )

    output_dir = Path(infer_config.get("output_dir", "outputs/infer"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(infer_config.get("seed", 7)))

    batch_size = int(infer_config.get("batch_size", 4))
    _, _, dataset_config, dataloader = load_dataset_bundle(
        args.data_config,
        batch_size=batch_size,
        shuffle=False,
        manifest_path=args.manifest_path,
    )
    batch = next(iter(dataloader))

    device = resolve_device(str(infer_config.get("device", "auto")))
    checkpoint_path = args.checkpoint or str(infer_config.get("checkpoint_path", ""))
    _, model = load_planner_model(
        args.model_config,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    batch = batch.to(device)
    num_samples = int(infer_config.get("num_samples", 3))
    predictions = model.sample(batch, num_samples=num_samples)
    scored = score_trajectory_candidates(
        predicted_samples=predictions,
        scene_batch=batch,
        time_delta=dataset_config.time_delta,
    )

    prediction_path = output_dir / "predictions.pt"
    torch.save(predictions.cpu(), prediction_path)
    plot_trajectory_comparison(
        predicted=scored["selected_trajectories"],
        target=batch.future_ego_trajectory,
        route_polylines=batch.route_lanes,
        route_mask=batch.route_lanes_mask,
        output_path=output_dir / "prediction_plot.png",
    )
    print(f"saved predictions to {prediction_path}")


if __name__ == "__main__":
    main()
