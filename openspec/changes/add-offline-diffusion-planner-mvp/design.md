## Context

The target system is a neural planner for autonomous driving. The source material and intended architecture point toward a diffusion-based planner that consumes multi-modal scene context and outputs a future ego trajectory. However, the full end-state also includes reward computation, reinforcement learning, and simulator integration, which would be too broad for a first implementation.

This change therefore defines the first practical slice: an offline diffusion planner MVP that can be trained and evaluated on logged data without any simulator dependency.

## Goals / Non-Goals

- Goals:
  - define a clean planner-first architecture for `nn_planner`
  - support canonical scene ingestion for ego, neighbors, lanes, and route context
  - train a conditional diffusion model to generate future ego trajectories
  - provide reproducible open-loop inference, metrics, and visualization
  - preserve extension points for reward and RL work

- Non-Goals:
  - no reinforcement learning policy optimization in this change
  - no C++ reward engine or pybind11 bridge in this change
  - no simulator rollout or closed-loop benchmark in this change
  - no attempt to support every possible dataset format in the core planner package

## Decisions

- Decision: keep the first implementation offline-only
  - Reason: the planner core must be inspectable and debuggable before adding reward and simulator feedback loops.

- Decision: use a canonical scene schema between data adapters and the planner
  - Reason: this prevents dataset-specific parsing logic from leaking into the model code and makes later dataset additions cheaper.

- Decision: separate preprocessing, encoding, diffusion decoding, training, and evaluation into different modules
  - Reason: the project will otherwise become tightly coupled around one training script and be hard to extend toward RL.

- Decision: represent heading internally with continuous features such as `cos/sin`
  - Reason: this avoids angle discontinuity issues and matches the documented planner behavior.

- Decision: anchor the first predicted trajectory step to the current ego state during inference
  - Reason: the generated future must start from the real current state rather than drift at the first step.

- Decision: make the MVP require iterative denoising but not require an optimized solver such as DPM-Solver on day one
  - Reason: correctness and modularity are more important than sampling speed in the first implementation. Faster samplers can be added later behind the same sampler interface.

## Proposed Architecture

```text
dataset adapter
  -> canonical scene tensors
  -> planner preprocess / normalization
  -> scene encoder
  -> conditional diffusion decoder
  -> sampled future trajectories
  -> metrics / visualization
```

Suggested module boundaries:

- `planner/datasets/`
  - dataset adapters and canonical sample objects
- `planner/preprocess/`
  - normalization, coordinate transforms, angle encoding, masks
- `planner/models/`
  - encoder, decoder, timestep embedding, mixer/attention blocks
- `planner/diffusion/`
  - schedule, sampler, noise utilities
- `planner/trainers/`
  - training loop, checkpointing, seed control
- `planner/metrics/`
  - open-loop metrics and route-consistency checks
- `planner/visualization/`
  - scene and trajectory debugging plots

## Risks / Trade-offs

- Risk: dataset adapters may disagree on field semantics
  - Mitigation: freeze a canonical scene contract early and keep adapters thin.

- Risk: the first architecture may overfit one scene representation
  - Mitigation: keep preprocessing contracts explicit and isolate dataset-specific logic outside core model modules.

- Risk: diffusion sampling may be too slow for large-scale experiments
  - Mitigation: introduce a sampler interface now so DPM-Solver or DDIM-style acceleration can be added later without rewriting the decoder.

- Risk: without reward modeling, open-loop metrics may hide closed-loop weaknesses
  - Mitigation: make this limitation explicit and preserve a clean seam for the next RL and reward proposal.

## Migration Plan

1. Create the project scaffold and canonical scene schema.
2. Implement preprocessing and normalization.
3. Implement the encoder and diffusion decoder.
4. Add training, inference, and evaluation scripts.
5. Add tests and debugging visualizations.
6. Create a follow-up proposal for reward and RL only after the offline planner is stable.

## Open Questions

- Which dataset should serve as the first adapter: nuPlan-style planning scenes, motion forecasting scenes, or an internal custom export?
- Should the first decoder backbone be a simpler U-Net-style 1D model or a Transformer-style DiT backbone from the start?
- Which open-loop metrics are required for MVP acceptance beyond displacement error?
