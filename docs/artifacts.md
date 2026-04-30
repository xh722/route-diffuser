# RouteDiffuser Artifacts

This document explains the main files written by the project and what each artifact is for.

## Output Roots

Default output directories:

- `outputs/manifests/`
- `outputs/stats/`
- `outputs/datasets/`
- `outputs/onnx/`
- `outputs/benchmarks/`
- `outputs/rollout/`
- `outputs/train/`
- `outputs/infer/`
- `outputs/eval/`
- `outputs/portfolio_demo/`

## Dataset Manifests

Location:

- `outputs/manifests/*.json`

Produced by:

- `python scripts/prepare_dataset.py`

Purpose:

- defines a stable subset or split for adapter-backed datasets
- decouples data selection from direct runtime sampling
- will become more important once non-synthetic adapters are added

Typical contents:

- dataset name
- adapter type
- split
- scene count
- per-entry ids, scenario names, sample indices, tags, metadata

## Dataset Statistics

Location:

- `outputs/stats/*.json`

Produced by:

- `python scripts/compute_dataset_stats.py`

Purpose:

- caches reusable per-feature summary statistics
- provides a stable inspection artifact for dataset ranges and scale
- lays the groundwork for future normalization reuse

Typical contents:

- dataset name and dataset type
- scene count
- per-tensor feature counts
- mean, std, min, and max for canonical tensor groups

## Exported NPZ Datasets

Location:

- `outputs/datasets/*.npz`

Produced by:

- `python scripts/export_dataset_npz.py`

Purpose:

- serializes canonical scene tensors into a public bridge format
- enables adapter roundtrip testing outside the synthetic generator
- provides a simple portable format for future public dataset conversions

## Training Artifacts

Location:

- `outputs/train/`

Produced by:

- `python scripts/train_planner.py`

Main files:

- `latest.pt`
- `train_log.csv`

Purpose:

- `latest.pt`: latest model checkpoint, including model state and optimizer state
- `train_log.csv`: epoch-level loss log for quick inspection

## Inference Artifacts

Location:

- `outputs/infer/`

Produced by:

- `python scripts/infer_planner.py`

Main files:

- `predictions.pt`
- `prediction_plot.png`

Purpose:

- `predictions.pt`: raw sampled trajectory tensor bundle
- `prediction_plot.png`: a quick visual comparison of selected prediction vs target

## ONNX Export Artifacts

Location:

- `outputs/onnx/`

Produced by:

- `python scripts/export_onnx.py`

Main files:

- `planner_denoiser.onnx`
- `planner_denoiser.json`
- `parity_report.json`

Purpose:

- `planner_denoiser.onnx`: deployable tensor-only denoiser core
- `planner_denoiser.json`: input/output shape and export metadata for the ONNX graph
- `parity_report.json`: tolerance-based comparison between PyTorch and ONNX outputs

## Benchmark Artifacts

Location:

- `outputs/benchmarks/`

Produced by:

- `python scripts/benchmark_infer.py`

Main files:

- `denoiser_core_benchmark.json`

Purpose:

- stores lightweight latency and throughput measurements for the PyTorch denoiser core
- gives a stable machine-readable artifact for future perf tracking

## Rollout Artifacts

Location:

- `outputs/rollout/`

Produced by:

- `python scripts/rollout_planner.py`

Main files:

- `rollout_trace.pt`
- `rollout_summary.json`
- `rollout_summary.md`
- `rollout_plot.png`

Purpose:

- `rollout_trace.pt`: raw executed trajectory trace, selected candidate indices, and collision flags
- `rollout_summary.json/.md`: lightweight closed-loop metrics and artifact pointers
- `rollout_plot.png`: executed path against the route and optional reference path

## Scorer Comparison Artifacts

Location:

- `outputs/eval/`

Produced by:

- `python scripts/compare_scorer.py`

Main files:

- `scorer_comparison.json`

Purpose:

- compares `heuristic` and `hybrid` selection under the same evaluation seed
- reports metric deltas attributable to the learned scorer path

## Ablation Matrix Artifacts

Location:

- `outputs/ablations/`

Produced by:

- `python scripts/run_ablation_matrix.py`

Main files:

- `scorer_ablation_matrix.json`
- `scorer_ablation_matrix.md`
- `encoder_ablation_matrix.json`
- `encoder_ablation_matrix.md`

Purpose:

- aggregates standard config sets into one summary artifact
- provides baseline deltas against the configured ablation baseline

## Evaluation Artifacts

Location:

- `outputs/eval/`

Produced by:

- `python scripts/eval_planner.py`

Main files:

- `evaluation_report.json`
- `evaluation_report.md`

Purpose:

- `evaluation_report.json`: structured report for downstream tooling or post-processing
- `evaluation_report.md`: human-readable report summary for GitHub and review

The evaluation report currently captures:

- dataset metadata
- candidate-selection settings
- overall metrics
- candidate-set metrics
- scenario-level metrics
- artifact paths
- run metadata

## Portfolio Demo Artifacts

Location:

- `outputs/portfolio_demo/`

Produced by:

- `python scripts/demo_planner.py`

Main files:

- `demo_checkpoint.pt`
- `predictions.pt`
- `prediction_plot.png`
- `candidate_trajectories.png`
- `scenario_gallery.png`
- `evaluation_report.json`
- `evaluation_report.md`
- `portfolio_summary.json`
- `portfolio_summary.md`

Purpose:

- `demo_checkpoint.pt`: checkpoint from the small end-to-end demo run
- `predictions.pt`: raw sampled outputs from the preview batch
- `prediction_plot.png`: selected trajectory against target
- `candidate_trajectories.png`: candidate bundle visualization
- `scenario_gallery.png`: multiple scene panels for quick visual inspection
- `evaluation_report.json/.md`: formal evaluation output for the demo run
- `portfolio_summary.json/.md`: GitHub- and resume-friendly summary layer on top of the evaluation

## Recommended Usage

Use the artifacts in this order:

1. dataset manifest
2. training checkpoint and log
3. inference plot or prediction tensor
4. structured evaluation report
5. portfolio summary and gallery assets

That separation keeps debugging, benchmarking, and presentation concerns clean.
