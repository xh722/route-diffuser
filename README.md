# RouteDiffuser

> Autonomous driving trajectory planning with a route-conditioned diffusion policy.

[简体中文 README](README.zh-CN.md)

`RouteDiffuser` is the public-facing project name for the `nn_planner` codebase. The goal is to
show the planner core of an autonomous driving system in a form that reads well on GitHub and in a
resume: scene representation, conditional trajectory generation, evaluation, and visual artifacts.

![RouteDiffuser demo plot](outputs/portfolio_demo/prediction_plot.png)
![RouteDiffuser scenario gallery](outputs/portfolio_demo/scenario_gallery.png)

## Why This Repo Exists

Most autonomous driving projects cannot ship their real datasets, simulator stacks, or internal
evaluation infrastructure. This repository focuses on the part that can be shown clearly:

- canonical scene-schema design for ego, neighbors, lanes, route polylines, and masks
- a route-prior residual diffusion policy with a conditional 1D U-Net decoder
- optional multi-resolution pyramid noise inspired by a larger reference diffusion planner
- heuristic candidate selection over route adherence, clearance, and comfort signals
- structured synthetic driving scenes that still look like real planning tasks
- scenario-wise open-loop metrics, checkpointing, candidate trajectory plots, and scenario galleries

## What It Demonstrates

- Built a route-conditioned planner that denoises trajectory residuals around a route prior.
- Implemented DDPM-style training, iterative inference, and first-step anchoring to the current ego state.
- Scored multiple sampled plans with route, clearance, and comfort heuristics instead of defaulting to the first sample.
- Modeled keep-lane, lane-change-left, lane-change-right, and curved-road scenarios in a reusable synthetic generator.
- Packaged the project with training, inference, evaluation, and portfolio demo scripts.

## Quick Start

```bash
pip install -e .[dev]
python scripts/prepare_dataset.py --output outputs/manifests/route_diffuser_synthetic_train.json
python scripts/demo_planner.py
```

Installed command aliases are also available after `pip install -e .[dev]`:

- `route-diffuser-prepare`
- `route-diffuser-export-npz`
- `route-diffuser-export-onnx`
- `route-diffuser-check-onnx`
- `route-diffuser-benchmark`
- `route-diffuser-rollout`
- `route-diffuser-train`
- `route-diffuser-infer`
- `route-diffuser-eval`
- `route-diffuser-demo`

Full command reference: [`docs/commands.md`](docs/commands.md)
Artifact reference: [`docs/artifacts.md`](docs/artifacts.md)
Architecture reference: [`docs/architecture.md`](docs/architecture.md)
Release checklist: [`docs/release_checklist.md`](docs/release_checklist.md)

That command produces a compact project showcase in `outputs/portfolio_demo/`:

- `prediction_plot.png`
- `candidate_trajectories.png`
- `scenario_gallery.png`
- `evaluation_report.json`
- `evaluation_report.md`
- `portfolio_summary.json`
- `portfolio_summary.md`
- `demo_checkpoint.pt`
- `predictions.pt`

## Command Surface

Public entry points:

- `python scripts/prepare_dataset.py`
- `python scripts/train_planner.py`
- `python scripts/infer_planner.py`
- `python scripts/eval_planner.py`
- `python scripts/demo_planner.py`

Compatibility wrappers:

- `python scripts/train_diffusion.py`
- `python scripts/infer_scene.py`
- `python scripts/eval_diffusion.py`
- `python scripts/demo_portfolio.py`

## Demo Outputs

The portfolio demo trains a small planner, samples future trajectories, evaluates open-loop metrics,
and writes a summary with scenario breakdowns, oracle candidate metrics, and selection diagnostics
that is easy to reuse in a GitHub project page or resume portfolio.

- Delivery roadmap: [`ROADMAP.md`](ROADMAP.md)
- Dataset preparation: [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py)
- Main demo script: [`scripts/demo_planner.py`](scripts/demo_planner.py)
- Generated summary: [`outputs/portfolio_demo/portfolio_summary.md`](outputs/portfolio_demo/portfolio_summary.md)
- Generated JSON: [`outputs/portfolio_demo/portfolio_summary.json`](outputs/portfolio_demo/portfolio_summary.json)
- Evaluation report: [`outputs/portfolio_demo/evaluation_report.md`](outputs/portfolio_demo/evaluation_report.md)
- Candidate trajectories: [`outputs/portfolio_demo/candidate_trajectories.png`](outputs/portfolio_demo/candidate_trajectories.png)
- Scenario gallery: [`outputs/portfolio_demo/scenario_gallery.png`](outputs/portfolio_demo/scenario_gallery.png)
- Legacy script names such as `demo_portfolio.py`, `train_diffusion.py`, `infer_scene.py`, and `eval_diffusion.py` remain as compatibility wrappers.

## Architecture

```text
structured scenario generator / future dataset adapter
  -> canonical scene tensors
  -> scene encoder
  -> route prior construction
  -> conditional diffusion decoder (1D U-Net)
  -> iterative trajectory denoising
  -> candidate scoring and selection
  -> open-loop metrics and debugging plots
```

## Repository Tour

- `planner/datasets/`: canonical scene schema and structured synthetic scenarios
- `planner/datasets/adapters/`: pluggable dataset adapters including synthetic and NPZ-backed public format loading
- `planner/models/`: scene encoder and conditional diffusion decoder
- `planner/diffusion/`: noise schedule and reverse diffusion utilities
- `planner/inference/`: anchoring plus heuristic candidate scoring
- `planner/trainers/`: training and evaluation loops
- `planner/visualization/`: trajectory plotting utilities
- `scripts/`: dataset preparation, train, infer, evaluate, and portfolio demo entry points
- `docs/commands.md`: public command manual
- `docs/artifacts.md`: output artifact reference
- `docs/architecture.md`: module and boundary overview
- `docs/release_checklist.md`: release-readiness checklist
- `ROADMAP.md`: execution plan for completing the public project

## Resume-Friendly Project Framing

- Autonomous driving planner focused on route-conditioned future trajectory generation.
- Conditional diffusion model with a 1D U-Net decoder implemented in PyTorch.
- Canonical scene interfaces, synthetic scenario generation, candidate ranking, scenario-level metrics, and visualization included.

## Scope

This repo is intentionally planner-first.

- included: scene modeling, diffusion planning, synthetic scenario generation, candidate scoring, evaluation, visualization, portfolio artifacts
- deferred: proprietary dataset adapters, simulator integration, reward modeling, RL fine-tuning

That tradeoff makes the codebase strong as a public project: it shows system design and modeling
ability without depending on private infrastructure.
