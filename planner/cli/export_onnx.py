"""Public ONNX export entry point for RouteDiffuser."""

from __future__ import annotations

import argparse
from pathlib import Path

from planner.cli.common import load_dataset_bundle, load_planner_model, resolve_device
from planner.common import load_yaml_config, set_seed
from planner.export import (
    OnnxExportConfig,
    build_export_metadata,
    export_planner_core_to_onnx,
    save_export_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/base.yaml")
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--inference-config", default="configs/inference/base.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output", default="outputs/onnx/planner_denoiser.onnx")
    parser.add_argument("--metadata-output", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--static-batch", action="store_true")
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

    device_name = args.device or str(infer_config.get("device", "auto"))
    device = resolve_device(device_name)
    checkpoint_path = args.checkpoint or str(infer_config.get("checkpoint_path", ""))
    _, model = load_planner_model(
        args.model_config,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    model = model.eval()
    scene_batch = scene_batch.to(device)

    export_config = OnnxExportConfig(
        opset_version=args.opset_version,
        dynamic_batch=not args.static_batch,
    )
    output_path = export_planner_core_to_onnx(
        model,
        scene_batch,
        args.output,
        config=export_config,
    )

    metadata = build_export_metadata(model, scene_batch, export_config)
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output_path.with_suffix(".json")
    )
    save_export_metadata(metadata, metadata_path)

    print(f"saved onnx model to {output_path}")
    print(f"saved export metadata to {metadata_path}")


if __name__ == "__main__":
    main()
