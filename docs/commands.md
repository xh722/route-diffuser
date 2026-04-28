# RouteDiffuser Commands

This document describes the public command surface for `RouteDiffuser`.

## Entry Points

You can invoke the project in two equivalent ways.

Script entry points:

- `python scripts/prepare_dataset.py`
- `python scripts/train_planner.py`
- `python scripts/infer_planner.py`
- `python scripts/eval_planner.py`
- `python scripts/demo_planner.py`

Installed command aliases after `pip install -e .[dev]`:

- `route-diffuser-prepare`
- `route-diffuser-train`
- `route-diffuser-infer`
- `route-diffuser-eval`
- `route-diffuser-demo`

Legacy compatibility wrappers still exist:

- `python scripts/train_diffusion.py`
- `python scripts/infer_scene.py`
- `python scripts/eval_diffusion.py`
- `python scripts/demo_portfolio.py`

## Shared Config Files

Default config files:

- data: `configs/data/synthetic.yaml`
- model: `configs/model/base.yaml`
- train: `configs/train/base.yaml`
- inference: `configs/inference/base.yaml`

The public CLI is built so these configs remain the primary control surface. Command-line flags are
mainly for small runtime overrides.

## Prepare Dataset

Generate a manifest for adapter-backed datasets.

Minimal example:

```bash
python scripts/prepare_dataset.py \
  --data-config configs/data/synthetic.yaml \
  --output outputs/manifests/route_diffuser_synthetic_train.json
```

Useful flags:

- `--split`: label the manifest split, such as `train`, `val`, `test`
- `--start-index`: start index for a subset manifest
- `--num-samples`: number of entries to include

Example subset manifest:

```bash
python scripts/prepare_dataset.py \
  --output outputs/manifests/tiny_train.json \
  --split train \
  --start-index 0 \
  --num-samples 8
```

Output:

- a JSON manifest under the requested path

## Train Planner

Train the planner and write checkpoints plus a CSV training log.

Minimal example:

```bash
python scripts/train_planner.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --train-config configs/train/base.yaml
```

Useful flags:

- `--manifest-path`: train from a prepared subset manifest
- `--output-dir`: override `outputs/train`
- `--epochs`: quick override without editing YAML
- `--batch-size`: quick override without editing YAML
- `--device`: `cpu`, `cuda`, or `auto`

Low-load example:

```bash
python scripts/train_planner.py \
  --manifest-path outputs/manifests/tiny_train.json \
  --epochs 1 \
  --batch-size 2 \
  --device cpu
```

Outputs:

- `train_log.csv`
- `latest.pt`

## Infer Planner

Sample future trajectories for one evaluation batch and write prediction artifacts.

Minimal example:

```bash
python scripts/infer_planner.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --inference-config configs/inference/base.yaml
```

Useful flags:

- `--checkpoint`: load a trained checkpoint
- `--manifest-path`: infer on a prepared manifest subset
- `--output-dir`: override `outputs/infer`
- `--batch-size`: override eval batch size
- `--num-samples`: candidate samples per scene
- `--device`: `cpu`, `cuda`, or `auto`

Outputs:

- `predictions.pt`
- `prediction_plot.png`

## Evaluate Planner

Run structured open-loop evaluation and emit formal JSON plus Markdown reports.

Minimal example:

```bash
python scripts/eval_planner.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --inference-config configs/inference/base.yaml
```

Useful flags:

- `--checkpoint`: evaluate a trained checkpoint
- `--manifest-path`: evaluate a prepared subset
- `--output-dir`: override `outputs/eval`
- `--batch-size`: override eval batch size
- `--num-samples`: candidate samples per scene
- `--device`: `cpu`, `cuda`, or `auto`

Outputs:

- `evaluation_report.json`
- `evaluation_report.md`

## Demo Planner

Run the full portfolio demo: train a small model, sample trajectories, generate visualizations, and
write a portfolio summary.

Minimal example:

```bash
python scripts/demo_planner.py
```

Useful flags:

- `--manifest-path`: run the demo on a prepared subset
- `--output-dir`: override `outputs/portfolio_demo`
- `--epochs`: shorten or lengthen the demo train run
- `--train-batch-size`: override training batch size
- `--eval-batch-size`: override evaluation batch size
- `--num-samples`: candidate samples per scene
- `--device`: `cpu`, `cuda`, or `auto`

Low-load example:

```bash
python scripts/demo_planner.py \
  --manifest-path outputs/manifests/tiny_train.json \
  --epochs 1 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --num-samples 2 \
  --device cpu
```

Outputs:

- `demo_checkpoint.pt`
- `predictions.pt`
- `prediction_plot.png`
- `candidate_trajectories.png`
- `scenario_gallery.png`
- `evaluation_report.json`
- `evaluation_report.md`
- `portfolio_summary.json`
- `portfolio_summary.md`

## Resource Guidance

If the machine is resource-constrained:

- generate a small manifest with `--num-samples`
- use `--device cpu` for smoke runs
- keep `--epochs` low for demo and training commands
- reduce `--batch-size`, `--train-batch-size`, and `--eval-batch-size`
- reduce `--num-samples` during inference and evaluation

The fastest low-risk smoke path is usually:

1. `prepare_dataset.py` with a tiny subset
2. `train_planner.py` with `--epochs 1 --batch-size 2 --device cpu`
3. `infer_planner.py` or `eval_planner.py` on the same manifest

## Artifact Locations

Default output roots:

- manifests: `outputs/manifests/`
- training: `outputs/train/`
- inference: `outputs/infer/`
- evaluation: `outputs/eval/`
- portfolio demo: `outputs/portfolio_demo/`
