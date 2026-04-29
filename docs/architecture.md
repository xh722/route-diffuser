# RouteDiffuser Architecture

This document describes how the major modules in `RouteDiffuser` fit together.

## Design Goal

The repository is organized around one core idea:

- keep the planner model, data interfaces, evaluation, export, and rollout boundaries explicit
- avoid mixing private-system assumptions into the public project structure

That means the code is intentionally split into small, named layers instead of one large training
script or one monolithic planner module.

## Top-Level Flow

```text
dataset config / manifest
  -> dataset adapter
  -> canonical scene tensors
  -> planner model
  -> candidate scoring
  -> open-loop evaluation / rollout / export
  -> reports and visual artifacts
```

## Module Layout

### `planner/datasets/`

Responsibility:

- define the canonical scene schema used everywhere else
- load or synthesize planning scenes through adapter-backed datasets
- keep data selection separate from model logic

Key pieces:

- `schema.py`: canonical tensor contract
- `adapters/`: pluggable dataset backends
- `factory.py`: config-driven dataset construction
- `synthetic.py`: backwards-compatible dataset wrapper and collate logic

Why it matters:

- future data adapters can be added without rewriting training or evaluation code
- manifests provide a stable split/subset selection layer

### `planner/models/`

Responsibility:

- implement the route-conditioned diffusion planner
- convert canonical scene tensors into denoised trajectory residual predictions

Key pieces:

- `scene_encoder.py`: scene context encoding
- `diffusion_decoder.py`: denoiser backbone
- `diffusion_planner.py`: assembled planner model and sampling logic

Why it matters:

- the repo can change the model internals later without changing the public data boundary

### `planner/inference/`

Responsibility:

- inference-time postprocessing and candidate selection

Key pieces:

- `anchoring.py`: first-step anchoring
- `scoring.py`: heuristic candidate ranking

Why it matters:

- separates model prediction from planning-time selection policy
- creates a clean future insertion point for learned scorers or value heads

### `planner/metrics/`

Responsibility:

- compute trajectory quality metrics for open-loop planning evaluation

Key pieces:

- `trajectory.py`: displacement, route, comfort, clearance, and candidate-set metrics

Why it matters:

- metrics remain reusable across evaluation, reports, and future regression checks

### `planner/reports/`

Responsibility:

- define stable serialized report contracts

Key pieces:

- `evaluation.py`: structured evaluation report schema

Why it matters:

- scripts stop inventing ad hoc JSON formats
- downstream tooling can rely on stable keys

### `planner/rollout/`

Responsibility:

- provide a lightweight closed-loop replanning loop

Key pieces:

- `simulator.py`: minimal rollout trace generation and summary logic

Why it matters:

- adds system-level behavior checks without requiring a heavy simulator service

### `planner/export/`

Responsibility:

- provide deployment-facing wrappers around the planner core

Key pieces:

- `onnx.py`: ONNX export wrapper for the tensor-only denoiser core
- `parity.py`: PyTorch vs ONNX parity reporting
- `benchmark.py`: small denoiser-core benchmark reporting

Why it matters:

- export, parity, and benchmark concerns stay decoupled from training code

### `planner/visualization/`

Responsibility:

- generate debugging and portfolio-facing plots

Key pieces:

- `trajectory.py`: single-scene, candidate, gallery, and rollout trace plots

Why it matters:

- visual outputs are treated as first-class artifacts, not notebook leftovers

### `planner/cli/`

Responsibility:

- hold the implementation for public command entry points

Key pieces:

- `common.py`: shared config, dataset, and model loading logic
- task-specific modules for prepare/export/train/infer/eval/demo/rollout

Why it matters:

- public scripts stay thin wrappers
- runtime behavior is centralized and easier to maintain

### `scripts/`

Responsibility:

- provide stable user-facing executable entry points

Why it matters:

- repository users can run commands directly without importing package modules
- legacy script names can remain as wrappers while public names stay clean

## Canonical Scene Contract

The canonical scene tensor contract is the most important interface in the project.

A valid scene batch includes:

- `ego_current_state`
- `neighbor_history`
- `neighbor_history_mask`
- `lane_polylines`
- `lane_polylines_mask`
- `route_lanes`
- `route_lanes_mask`
- optional future trajectory tensors for supervised training and evaluation

Everything else in the project assumes this shape contract is stable.

## Export Boundary

The deployment-facing export boundary is intentionally narrower than the full planner.

Current ONNX export covers:

- canonical scene tensors
- noisy trajectory residual input
- timestep input
- predicted noise output

It does not export:

- the full iterative DDPM sampling loop
- candidate scoring logic
- closed-loop rollout logic

This is deliberate. The denoiser core is the cleanest deployable unit.

## Rollout Boundary

The closed-loop rollout layer is intentionally lightweight.

Current rollout provides:

- receding-horizon replanning
- selected candidate execution
- route-relative trace plotting
- lightweight collision and route deviation summaries

It is not a full simulator platform. That boundary keeps the public project maintainable.

## Artifact Philosophy

The repo treats artifacts as explicit products of each layer:

- manifests for data
- checkpoints and logs for training
- plots and tensors for inference
- JSON/Markdown reports for evaluation
- ONNX and parity reports for deployment
- rollout traces and summaries for closed-loop analysis

This is what makes the repository feel like a complete engineering project instead of a notebook
collection.
