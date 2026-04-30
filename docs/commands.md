# RouteDiffuser Commands

This document describes the public command surface for `RouteDiffuser`.

## Entry Points

You can invoke the project in two equivalent ways.

Script entry points:

- `python scripts/prepare_dataset.py`
- `python scripts/compute_dataset_stats.py`
- `python scripts/export_dataset_npz.py`
- `python scripts/export_onnx.py`
- `python scripts/check_onnx_parity.py`
- `python scripts/benchmark_infer.py`
- `python scripts/rollout_planner.py`
- `python scripts/compare_scorer.py`
- `python scripts/run_ablation_matrix.py`
- `python scripts/analyze_failures.py`
- `python scripts/train_planner.py`
- `python scripts/infer_planner.py`
- `python scripts/eval_planner.py`
- `python scripts/demo_planner.py`

Installed command aliases after `pip install -e .[dev]`:

- `route-diffuser-prepare`
- `route-diffuser-stats`
- `route-diffuser-export-npz`
- `route-diffuser-export-onnx`
- `route-diffuser-check-onnx`
- `route-diffuser-benchmark`
- `route-diffuser-rollout`
- `route-diffuser-compare-scorer`
- `route-diffuser-ablations`
- `route-diffuser-failures`
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
- data example for public NPZ format: `configs/data/npz_example.yaml`
- model: `configs/model/base.yaml`
- ablation config for heuristic-only selection: `configs/model/heuristic_only.yaml`
- ablation config for light learned-scorer influence: `configs/model/learned_scorer_light.yaml`
- ablation config for learned scorer experiments: `configs/model/learned_scorer.yaml`
- ablation config for stronger learned-scorer influence: `configs/model/learned_scorer_strong.yaml`
- encoder-scale ablations: `configs/model/encoder_small.yaml`, `configs/model/encoder_wide.yaml`, `configs/model/encoder_attention.yaml`
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

For NPZ-backed public-format data, point the data config at `configs/data/npz_example.yaml` and
set `source_path` to your normalized `.npz` file.

## Compute Dataset Statistics

Compute cached per-feature statistics for one dataset configuration or prepared subset.

Minimal example:

```bash
python scripts/compute_dataset_stats.py \
  --data-config configs/data/synthetic.yaml \
  --output outputs/stats/route_diffuser_synthetic_stats.json
```

Useful flags:

- `--manifest-path`: compute stats for a prepared subset
- `--batch-size`: CPU-side aggregation batch size
- `--max-scenes`: limit the number of scenes used for the cache
- `--output`: target JSON cache path

Typical use:

1. prepare a subset manifest if needed
2. compute and save dataset statistics once
3. reuse the cached stats file as a reference artifact for normalization and dataset inspection

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

## Export Dataset To NPZ

Export canonical planning samples into the public `.npz` bridge format.

Minimal example:

```bash
python scripts/export_dataset_npz.py \
  --data-config configs/data/synthetic.yaml \
  --output outputs/datasets/route_diffuser_synthetic.npz
```

Useful flags:

- `--manifest-path`: export only a prepared subset
- `--output`: target `.npz` file

Typical use:

1. generate a subset manifest
2. export that subset to `.npz`
3. point `configs/data/npz_example.yaml` at the exported file
4. run `prepare_dataset.py` again on the NPZ-backed config if needed

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

## Export ONNX

Export the tensor-only planner denoiser core to ONNX.

Minimal example:

```bash
python scripts/export_onnx.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --output outputs/onnx/planner_denoiser.onnx
```

Useful flags:

- `--checkpoint`: export a trained checkpoint
- `--manifest-path`: export using a prepared subset
- `--output`: target ONNX file path
- `--metadata-output`: target JSON metadata path
- `--batch-size`: example batch used for tracing
- `--device`: `cpu`, `cuda`, or `auto`
- `--opset-version`: ONNX opset version
- `--static-batch`: disable dynamic batch axes

Important note:

- this command exports the denoiser core, not the full iterative DDPM sampling loop
- the exported graph corresponds to:
  canonical scene tensors + noisy trajectory + timestep -> predicted noise

## Check ONNX Parity

Compare the exported ONNX denoiser core against the PyTorch wrapper on one batch.

Minimal example:

```bash
python scripts/check_onnx_parity.py \
  --onnx-path outputs/onnx/planner_denoiser.onnx \
  --data-config configs/data/synthetic.yaml
```

Useful flags:

- `--checkpoint`: parity-check a trained checkpoint
- `--manifest-path`: parity-check on a prepared subset
- `--output`: target JSON report path
- `--batch-size`: parity-check batch size
- `--device`: torch-side device, recommended `cpu` for light checks
- `--atol`: absolute tolerance
- `--rtol`: relative tolerance

Output:

- `parity_report.json`

This command requires `onnxruntime`.

## Benchmark Denoiser Core

Run a lightweight latency benchmark for the PyTorch denoiser core.

Minimal example:

```bash
python scripts/benchmark_infer.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --output outputs/benchmarks/denoiser_core_benchmark.json \
  --device cpu
```

Useful flags:

- `--checkpoint`: benchmark a trained checkpoint
- `--manifest-path`: benchmark a prepared subset
- `--output`: target JSON report path
- `--batch-size`: benchmark batch size
- `--warmup-iterations`: warmup count before timing
- `--iterations`: measured iterations
- `--device`: `cpu`, `cuda`, or `auto`

Output:

- `denoiser_core_benchmark.json`

This benchmark only measures the denoiser core forward path, not the full iterative sampling loop.

## Closed-Loop Rollout

Run a lightweight receding-horizon rollout on one scenario.

Minimal example:

```bash
python scripts/rollout_planner.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --output-dir outputs/rollout \
  --device cpu
```

Useful flags:

- `--checkpoint`: rollout a trained checkpoint
- `--manifest-path`: rollout a prepared subset
- `--output-dir`: target rollout artifact directory
- `--num-steps`: rollout horizon in replanning steps
- `--num-samples`: candidate samples per step
- `--device`: `cpu`, `cuda`, or `auto`

Outputs:

- `rollout_trace.pt`
- `rollout_summary.json`
- `rollout_summary.md`
- `rollout_plot.png`

This is a lightweight closed-loop planner loop, not a full simulator service.

## Compare Scorer Modes

Compare `heuristic` and `hybrid` candidate selection under the same evaluation seed.

Minimal example:

```bash
python scripts/compare_scorer.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/learned_scorer.yaml \
  --output outputs/eval/scorer_comparison.json \
  --device cpu
```

Useful flags:

- `--checkpoint`: compare a trained checkpoint
- `--manifest-path`: compare on a prepared subset
- `--output`: target JSON report path
- `--batch-size`: evaluation batch size
- `--num-samples`: candidate samples per scene
- `--device`: recommended `cpu` for light checks

Output:

- `scorer_comparison.json`

## Run Ablation Matrix

Run the standard scorer ablation config set and summarize all runs into one matrix.

Minimal example:

```bash
python scripts/run_ablation_matrix.py \
  --data-config configs/data/synthetic.yaml \
  --output-dir outputs/ablations/scorer_matrix \
  --device cpu
```

Default config set:

- `configs/model/heuristic_only.yaml`
- `configs/model/learned_scorer_light.yaml`
- `configs/model/learned_scorer.yaml`
- `configs/model/learned_scorer_strong.yaml`

Useful flags:

- `--matrix scorer`
- `--matrix encoder`
- `--model-configs ...` to override the preset config set
- `--baseline-config ...` to override the baseline

Encoder matrix example:

```bash
python scripts/run_ablation_matrix.py \
  --matrix encoder \
  --output-dir outputs/ablations/encoder_matrix \
  --device cpu
```

Outputs:

- `scorer_ablation_matrix.json/.md`
- `encoder_ablation_matrix.json/.md`

This is the preferred command when you want one compact summary instead of multiple manual compare
invocations.

## Analyze Failures

Rank the worst scenes by one chosen metric and produce a failure-analysis report.

Minimal example:

```bash
python scripts/analyze_failures.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --output-dir outputs/eval/failures \
  --ranking-metric fde \
  --device cpu
```

Useful flags:

- `--selection-mode`: `auto`, `heuristic`, or `hybrid`
- `--ranking-metric`: for example `fde`, `ade`, `route_error`
- `--top-k`: number of worst cases to retain overall and per scenario
- `--manifest-path`: restrict analysis to a subset
- `--checkpoint`: analyze a trained checkpoint

Outputs:

- `failure_analysis.json`
- `failure_analysis.md`

Recommended ablation pair:

- `configs/model/heuristic_only.yaml`
- `configs/model/learned_scorer.yaml`

Recommended multi-level ablation:

- `configs/model/heuristic_only.yaml`
- `configs/model/learned_scorer_light.yaml`
- `configs/model/learned_scorer.yaml`
- `configs/model/learned_scorer_strong.yaml`

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

## Public NPZ Format

The first public-format adapter uses one `.npz` file containing canonical scene tensors.

Required keys:

- `ego_current_state`
- `neighbor_history`
- `neighbor_history_mask`
- `lane_polylines`
- `lane_polylines_mask`
- `route_lanes`
- `route_lanes_mask`
- `future_ego_trajectory`
- `future_ego_mask`

Optional key:

- `scenario_name`

Expected leading batch dimension:

- each array should have shape `[N, ...]`
- all arrays must agree on `N`

This adapter is meant as a public bridge format, not as the final large-scale dataset interface.
