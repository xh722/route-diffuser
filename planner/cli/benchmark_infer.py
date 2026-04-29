"""Public benchmark entry point for the RouteDiffuser denoiser core."""

from __future__ import annotations

import argparse
from pathlib import Path

from planner.cli.common import load_dataset_bundle, load_planner_model, resolve_device
from planner.common import load_yaml_config, set_seed
from planner.export import BenchmarkConfig, benchmark_torch_denoiser_core, save_benchmark_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/base.yaml")
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--inference-config", default="configs/inference/base.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output", default="outputs/benchmarks/denoiser_core_benchmark.json")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_config = load_yaml_config(args.inference_config)
    set_seed(int(infer_config.get("seed", 7)))

    batch_size = args.batch_size or int(infer_config.get("batch_size", 4))
    _, _, _, dataloader = load_dataset_bundle(
        args.data_config,
        batch_size=batch_size,
        shuffle=False,
        manifest_path=args.manifest_path,
    )
    scene_batch = next(iter(dataloader))

    checkpoint_path = args.checkpoint or str(infer_config.get("checkpoint_path", ""))
    device_name = args.device or str(infer_config.get("device", "auto"))
    device = resolve_device(device_name)
    _, model = load_planner_model(
        args.model_config,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    scene_batch = scene_batch.to(device)

    report = benchmark_torch_denoiser_core(
        model,
        scene_batch,
        BenchmarkConfig(
            warmup_iterations=args.warmup_iterations,
            measure_iterations=args.iterations,
        ),
    )
    output_path = save_benchmark_report(report, Path(args.output))

    print(f"saved benchmark report to {output_path}")
    print(
        f"mean_latency_ms={report['mean_latency_ms']:.6f} throughput={report['throughput_samples_per_sec']:.3f}"
    )


if __name__ == "__main__":
    main()
