"""Public failure-analysis entry point for RouteDiffuser."""

from __future__ import annotations

import argparse
from pathlib import Path

from planner.cli.common import load_dataset_bundle, load_planner_model, resolve_device
from planner.common import load_yaml_config, set_seed
from planner.reports import analyze_failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/base.yaml")
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--inference-config", default="configs/inference/base.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output-dir", default="outputs/eval/failures")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selection-mode", default="auto")
    parser.add_argument("--ranking-metric", default="fde")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_config = load_yaml_config(args.inference_config)
    set_seed(int(infer_config.get("seed", 7)))

    batch_size = args.batch_size or int(infer_config.get("batch_size", 4))
    data_config, _, dataset_config, dataloader = load_dataset_bundle(
        args.data_config,
        batch_size=batch_size,
        shuffle=False,
        manifest_path=args.manifest_path,
    )
    checkpoint_path = args.checkpoint or str(infer_config.get("checkpoint_path", ""))
    device = resolve_device(args.device or "cpu")
    _, model = load_planner_model(
        args.model_config,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "failure_analysis.json"
    markdown_path = output_dir / "failure_analysis.md"

    report = analyze_failures(
        model=model,
        dataloader=dataloader,
        device=device,
        num_samples=args.num_samples or int(infer_config.get("num_samples", 3)),
        time_delta=dataset_config.time_delta,
        ranking_metric=args.ranking_metric,
        top_k=args.top_k,
        dataset_name=str(getattr(dataset_config, "dataset_name", "unknown_dataset")),
        dataset_type=str(data_config.get("dataset_type", "synthetic")),
        split="eval",
        selection_mode=args.selection_mode,
        artifacts={
            "json_report": str(json_path),
            "markdown_report": str(markdown_path),
        },
        metadata={
            "checkpoint_path": checkpoint_path,
            "device": str(device),
            "batch_size": batch_size,
        },
    )
    report.save_json(json_path)
    report.save_markdown(markdown_path)
    print(report.to_json())


if __name__ == "__main__":
    main()
