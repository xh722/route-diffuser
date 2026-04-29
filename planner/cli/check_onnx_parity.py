"""Public ONNX parity-check entry point for RouteDiffuser."""

from __future__ import annotations

import argparse
from pathlib import Path

from planner.cli.common import load_dataset_bundle, load_planner_model, resolve_device
from planner.common import load_yaml_config, set_seed
from planner.export import (
    OnnxParityConfig,
    build_parity_report,
    run_onnx_denoiser_core,
    run_torch_denoiser_core,
    save_parity_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--model-config", default="configs/model/base.yaml")
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--inference-config", default="configs/inference/base.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output", default="outputs/onnx/parity_report.json")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
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
    device = resolve_device(args.device or "cpu")
    _, model = load_planner_model(
        args.model_config,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    torch_output, onnx_inputs = run_torch_denoiser_core(model, scene_batch)
    onnx_output = run_onnx_denoiser_core(args.onnx_path, onnx_inputs)
    report = build_parity_report(
        torch_output,
        onnx_output,
        OnnxParityConfig(atol=args.atol, rtol=args.rtol),
        onnx_path=args.onnx_path,
    )
    output_path = save_parity_report(report, Path(args.output))

    print(f"saved parity report to {output_path}")
    print(f"passed={report['passed']} max_abs_error={report['max_abs_error']:.6e}")


if __name__ == "__main__":
    main()
