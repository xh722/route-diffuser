"""Public training entry point for RouteDiffuser."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from planner.cli.common import apply_overrides, load_dataset_bundle, load_planner_model, resolve_device
from planner.common import load_yaml_config, set_seed
from planner.trainers import create_optimizer, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/base.yaml")
    parser.add_argument("--data-config", default="configs/data/synthetic.yaml")
    parser.add_argument("--train-config", default="configs/train/base.yaml")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = load_yaml_config(args.train_config)
    train_config = apply_overrides(
        train_config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    output_dir = Path(train_config.get("output_dir", "outputs/train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(train_config.get("seed", 7)))

    batch_size = int(train_config.get("batch_size", 8))
    data_config, _, _, dataloader = load_dataset_bundle(
        args.data_config,
        batch_size=batch_size,
        shuffle=True,
        manifest_path=args.manifest_path,
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

    log_path = output_dir / "train_log.csv"
    checkpoint_path = output_dir / "latest.pt"
    epochs = int(train_config.get("epochs", 2))
    grad_clip_norm = float(train_config.get("grad_clip_norm", 1.0))

    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "num_batches"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            metrics = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                grad_clip_norm=grad_clip_norm,
            )
            writer.writerow({"epoch": epoch, **metrics})
            handle.flush()

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_config,
                    "data_config": data_config,
                    "train_config": train_config,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"epoch={epoch} loss={metrics['loss']:.6f}")

    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
