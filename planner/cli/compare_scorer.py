"""Compare heuristic and hybrid candidate selection on one evaluation pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planner.cli.common import load_dataset_bundle, load_planner_model, resolve_device
from planner.common import load_yaml_config, set_seed
from planner.trainers import evaluate_model_detailed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/base.yaml")
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--inference-config", default="configs/inference/base.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output", default="outputs/eval/scorer_comparison.json")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _metric_deltas(
    heuristic: dict[str, float],
    hybrid: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(heuristic) & set(hybrid))
    return {key: round(hybrid[key] - heuristic[key], 6) for key in keys}


def main() -> None:
    args = parse_args()
    infer_config = load_yaml_config(args.inference_config)
    seed = int(infer_config.get("seed", 7))

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

    common_kwargs = dict(
        model=model,
        dataloader=dataloader,
        device=device,
        num_samples=args.num_samples or int(infer_config.get("num_samples", 3)),
        time_delta=dataset_config.time_delta,
        dataset_name=str(getattr(dataset_config, "dataset_name", "unknown_dataset")),
        dataset_type=str(data_config.get("dataset_type", "synthetic")),
        split="eval",
        metadata={
            "checkpoint_path": checkpoint_path,
            "device": str(device),
            "batch_size": batch_size,
        },
    )

    set_seed(seed)
    heuristic_report = evaluate_model_detailed(
        **common_kwargs,
        selection_mode="heuristic",
    )
    set_seed(seed)
    hybrid_report = evaluate_model_detailed(
        **common_kwargs,
        selection_mode="hybrid",
    )

    payload = {
        "report_type": "scorer_mode_comparison",
        "dataset_name": common_kwargs["dataset_name"],
        "dataset_type": common_kwargs["dataset_type"],
        "selection_modes": {
            "heuristic": heuristic_report.selection.to_dict(),
            "hybrid": hybrid_report.selection.to_dict(),
        },
        "overall_metrics": {
            "heuristic": heuristic_report.overall_metrics,
            "hybrid": hybrid_report.overall_metrics,
            "delta_hybrid_minus_heuristic": _metric_deltas(
                heuristic_report.overall_metrics,
                hybrid_report.overall_metrics,
            ),
        },
        "candidate_set_metrics": {
            "heuristic": heuristic_report.candidate_set_metrics,
            "hybrid": hybrid_report.candidate_set_metrics,
            "delta_hybrid_minus_heuristic": _metric_deltas(
                heuristic_report.candidate_set_metrics,
                hybrid_report.candidate_set_metrics,
            ),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
