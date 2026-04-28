# Change: Add offline diffusion planner MVP

## Why

`nn_planner` is currently an empty project. We need a concrete first milestone that turns the planning idea into a buildable system with clear module boundaries, instead of trying to implement diffusion planning, reward modeling, reinforcement learning, and simulator integration all at once.

The first milestone should establish an offline route-conditioned diffusion planner that can train on logged scenes, generate future ego trajectories, and produce open-loop metrics and visualizations. This gives the project a usable core and a stable base for later reward and RL work.

## What Changes

- define the initial project context and capability boundaries for `nn_planner`
- add an offline diffusion planning MVP centered on canonical scene inputs, multi-modal scene encoding, and conditional trajectory generation
- add a training workflow covering preprocessing, normalization, diffusion-noise supervision, checkpointing, and evaluation artifacts
- require inference-time anchoring of the first ego state so generated trajectories start from the current vehicle state
- keep reinforcement learning, C++ reward acceleration, and closed-loop simulator rollout explicitly out of scope for this MVP

## Impact

- Affected specs:
  - `offline-diffusion-planning`
  - `planner-training-workflow`
- Affected code:
  - new `planner/` packages for preprocessing, models, diffusion utilities, training, evaluation, and visualization
  - new `configs/` and `scripts/` entries for training and inference
  - future test coverage in `tests/`
- Breaking changes: none, because this is an initial scaffold
