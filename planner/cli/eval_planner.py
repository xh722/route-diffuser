"""Public evaluation entry point for RouteDiffuser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planner.cli.common import apply_overrides, load_dataset_bundle, load_planner_model, resolve_device
from planner.common import load_yaml_config, set_seed
from planner.trainers import evaluate_model_detailed


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
    parser.add_argument("--selection-mode", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_config = load_yaml_config(args.inference_config)
    infer_config = apply_overrides(
        infer_config,
        eval_output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=args.device,
    )

    output_dir = Path(infer_config.get("eval_output_dir", "outputs/eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(infer_config.get("seed", 7)))

    batch_size = int(infer_config.get("batch_size", 4))
    data_config, _, dataset_config, dataloader = load_dataset_bundle(
        args.data_config,
        batch_size=batch_size,
        shuffle=False,
        manifest_path=args.manifest_path,
    )

    device = resolve_device(str(infer_config.get("device", "auto")))
    checkpoint_path = args.checkpoint or str(infer_config.get("checkpoint_path", ""))
    _, model = load_planner_model(
        args.model_config,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    report_json_path = output_dir / "evaluation_report.json"
    report_markdown_path = output_dir / "evaluation_report.md"
    report = evaluate_model_detailed(
        model=model,
        dataloader=dataloader,
        device=device,
        num_samples=int(infer_config.get("num_samples", 1)),
        time_delta=dataset_config.time_delta,
        dataset_name=str(getattr(dataset_config, "dataset_name", "unknown_dataset")),
        dataset_type=str(data_config.get("dataset_type", "synthetic")),
        split="eval",
        artifacts={
            "json_report": str(report_json_path),
            "markdown_report": str(report_markdown_path),
        },
        metadata={
            "checkpoint_path": checkpoint_path,
            "device": str(device),
            "batch_size": batch_size,
        },
        selection_mode=args.selection_mode,
    )
    report.save_json(report_json_path)
    report.save_markdown(report_markdown_path)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
