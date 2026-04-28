## 1. Project scaffold

- [x] 1.1 Create the initial `planner/`, `configs/`, `scripts/`, and `tests/` package layout described in the proposal
- [x] 1.2 Define a canonical scene input schema for ego, neighbors, lanes, route lanes, masks, and future labels
- [x] 1.3 Add structured configuration files for model, data, training, and inference settings

## 2. Offline planner core

- [x] 2.1 Implement preprocessing and normalization utilities, including local-frame conversion and heading `cos/sin` encoding
- [x] 2.2 Implement the scene encoder boundary for dynamic agents, lanes, and route context
- [x] 2.3 Implement a conditional diffusion decoder that predicts trajectory noise from scene context and diffusion timestep
- [x] 2.4 Enforce first-state anchoring during inference so generated trajectories begin from the current ego state

## 3. Training and inference workflow

- [x] 3.1 Add a training script that samples diffusion timesteps, injects noise into ground-truth future trajectories, and optimizes a noise-prediction loss
- [x] 3.2 Add checkpointing, seed control, and experiment logging
- [x] 3.3 Add an inference script that generates one or more future trajectory samples for a given scene batch

## 4. Evaluation and debugging

- [x] 4.1 Add open-loop metrics such as displacement error and route-consistency checks
- [x] 4.2 Add trajectory visualization for predicted vs ground-truth futures
- [x] 4.3 Add unit and integration tests for preprocessing, shape contracts, training smoke tests, and inference smoke tests

## 5. Deferred follow-up

- [x] 5.1 Document the extension seam for reward modeling and RL without implementing it in this change
- [x] 5.2 Document the extension seam for simulator adapters without implementing them in this change
