# RouteDiffuser Portfolio Case Study

This document frames `RouteDiffuser` as a resume-ready autonomous driving planning project.

## Project Summary

`RouteDiffuser` is a public planner-core project for route-conditioned autonomous driving trajectory
generation. It focuses on the parts of a planning stack that can be shared and evaluated without
private logs, simulator services, or internal deployment infrastructure:

- canonical scene tensors for ego state, neighbor history, lanes, route polylines, and masks
- route-prior residual diffusion for future trajectory generation
- multi-candidate scoring and selection instead of taking the first sampled trajectory
- scenario-level evaluation, failure analysis, ablation tooling, and visual artifacts
- ONNX export, parity checking, benchmarking, and lightweight closed-loop rollout

## Problem Framing

Many planning models produce a single future trajectory, but real planning systems usually evaluate
multiple candidates against route following, safety, comfort, and downstream feasibility. This repo
models that workflow in a compact, reproducible way:

```text
scene tensors
  -> route prior
  -> diffusion residual sampler
  -> candidate trajectory bundle
  -> heuristic / learned hybrid scoring
  -> structured reports, plots, rollout summaries
```

## Key Technical Decisions

- **Route-prior residual prediction**: The diffusion model denoises residuals around a route-derived
  trajectory prior, making route adherence a first-class part of generation rather than a post-hoc
  penalty only.
- **Canonical public schema**: The planner consumes a stable tensor contract, so synthetic data,
  NPZ bridge data, and future adapters can share the same training/evaluation path.
- **Candidate scoring layer**: Inference samples multiple plans and ranks them with route,
  clearance, comfort, vehicle-footprint collision, and optional learned scorer signals.
- **Route-anchor candidates**: The scorer can train against lane-intention style candidates such as
  route-centered, left/right lateral offsets, and faster/slower progress anchors.
- **Vehicle-footprint safety metrics**: Collision checks use oriented bounding boxes rather than only
  point-distance thresholds, and the same metric is used in open-loop evaluation and rollout.
- **Experiment-grade reports**: Evaluation produces JSON and Markdown reports with overall,
  candidate-set, scenario-level, scorer-diagnostic, and safety metrics.

## What To Show In An Interview

- `README.md`: high-level project entry point and generated visual artifacts
- `docs/architecture.md`: module boundaries and why each layer exists
- `docs/ablations.md`: scorer and encoder experiment axes
- `outputs/portfolio_demo/scenario_gallery.png`: scenario coverage visualization
- `outputs/portfolio_demo/candidate_trajectories.png`: multi-candidate planning behavior
- `outputs/portfolio_demo/portfolio_summary.md`: compact metric and artifact summary

## Suggested Resume Bullets

- Built a route-conditioned autonomous driving trajectory planner in PyTorch using a conditional
  diffusion policy and 1D U-Net denoising decoder.
- Designed a canonical planner scene schema covering ego state, neighbor history, lane polylines,
  route polylines, masks, and future trajectory targets.
- Implemented route-prior residual sampling, multi-candidate trajectory generation, hybrid
  heuristic/learned scoring, and route-anchor candidate supervision.
- Added planning evaluation infrastructure with ADE/FDE, route consistency, comfort proxies,
  oriented-box collision rate, candidate oracle metrics, scenario breakdowns, and failure analysis.
- Built reproducible public tooling for dataset manifests, NPZ bridge export, ONNX export/parity,
  CPU/GPU benchmarking, closed-loop rollout, ablation matrices, and portfolio visual artifacts.

## Interview Talking Points

- **Why diffusion?** It provides a natural way to sample multiple plausible futures, which makes
  candidate selection and uncertainty analysis visible.
- **Why route priors?** Route priors keep the generative model grounded in navigation intent and
  reduce the burden on pure learned trajectory generation.
- **Why not a full simulator?** The public repo intentionally avoids private simulator/service
  dependencies and instead implements a lightweight rollout loop that is understandable and
  reproducible.
- **How is safety measured?** The project reports both point-distance collision and oriented-box
  collision; box collision is the main metric because it better reflects vehicle footprint overlap.
- **How is the learned scorer debugged?** Reports include heuristic regret, learned preference
  regret, and agreement rates so scorer behavior can be evaluated beyond average ADE/FDE.

## Current Limitations

- Synthetic scenarios are useful for reproducibility, but real public driving datasets would provide
  stronger evidence of model quality.
- The ONNX path exports the denoiser core, not the full iterative DDPM sampling loop.
- The rollout simulator is intentionally lightweight and does not replace a full closed-loop
  autonomy simulator.
- The learned scorer is positioned as an experiment axis; it still needs calibration and stronger
  supervision before being treated as a production-quality selector.

## Next High-Value Extensions

- Add a documented public dataset example built from exported NPZ bridge files.
- Add a scorer calibration report that compares heuristic best, learned best, selected candidate,
  and oracle ADE/FDE ranks.
- Add richer failure-case galleries for route deviation, collision, and comfort violations.
- Add a small model-card style artifact describing training data, metrics, intended use, and limits.
