# RouteDiffuser Roadmap

This document turns the current `planner core` repository into a concrete delivery plan for a
public, reproducible autonomous driving planning project.

The intent is not to clone a private production stack. The intent is to translate the strongest
ideas from a larger planning system into a public repo that is actually finishable.

## Project Goal

Build `RouteDiffuser` into a complete public planning project with:

- reproducible training, inference, evaluation, and demo commands
- at least one non-synthetic data adapter
- formal evaluation reports instead of ad hoc metrics only
- deployable inference export via ONNX
- a lightweight closed-loop rollout environment
- documentation, tests, and release-ready artifacts

## Definition Of Done

`V1` is complete when all of the following are true:

1. A new machine can clone the repo, install dependencies, and run train, eval, infer, export,
   and demo commands from the README.
2. The repo supports both the current synthetic generator and at least one public-format dataset
   adapter.
3. Evaluation produces structured JSON and Markdown reports with overall metrics, scenario
   breakdowns, candidate-set metrics, and safety/comfort proxies.
4. The model can be exported to ONNX and checked for numerical parity against PyTorch inference.
5. The repo includes a lightweight closed-loop rollout demo with metrics and visual artifacts.
6. Tests and smoke checks cover the main pipeline and pass in CI.

## Non-Goals For V1

These are explicitly out of scope for the first complete version:

- private protobuf or service stacks
- proprietary vehicle logs or company-internal dataset formats
- TensorRT or private deployment plugins
- multi-node distributed training infrastructure
- production RL serving or reward-module bindings
- full simulator integration equivalent to internal autonomous driving platforms

## Gap Map

The current repo is already strong in the planner core:

- canonical scene schema
- diffusion planner model
- synthetic scenario generation
- open-loop evaluation
- candidate scoring heuristics
- portfolio-quality demo artifacts

The missing layers are what make the project feel complete:

| Capability | Current Repo | Needed For V1 |
| --- | --- | --- |
| Data preparation | Synthetic only | Public-format adapter, dataset manifests, preprocessing scripts |
| Evaluation | Open-loop metrics and reports | Formal report package, scenario taxonomies, benchmark outputs |
| Deployment | PyTorch only | ONNX export, parity tests, latency benchmark |
| Closed-loop behavior | None | Lightweight rollout simulator and closed-loop metrics |
| Experiment management | Basic configs | Structured runs, checkpoints, reproducible experiment table |
| Release packaging | README and demo | Command matrix, model card, roadmap, CI smoke checks |

## Delivery Strategy

The repo should evolve in this order:

1. Standardize interfaces and artifacts.
2. Add one realistic data path.
3. Add evaluation and deployment as first-class features.
4. Add a lightweight closed-loop environment.
5. Only then spend major effort on model upgrades.

This keeps the project from becoming a research sandbox with incomplete engineering.

## Milestones

### P0: Foundation And Public Project Skeleton

Focus: make the repo operational as a real project instead of only a demo.

Target outcome:

- one clear command path for synthetic training, inference, evaluation, and demo
- stable artifact layout under `outputs/`
- extensible dataset adapter boundary
- formal evaluation report schema

Planned work:

- [x] Add `planner/datasets/adapters/` with a stable adapter interface.
- [x] Add `scripts/prepare_dataset.py` for manifest generation and preprocessing.
- [x] Introduce dataset manifests such as JSON/JSONL/NPZ index files instead of direct ad hoc loading.
- [x] Separate synthetic dataset config from future real-data adapter configs.
- [x] Refactor evaluation output into a stable report contract:
      overall metrics, scenario metrics, oracle metrics, selection diagnostics, artifact paths.
- [x] Add smoke scripts for `train`, `infer`, `eval`, and `demo`.
- [x] Add a minimal CI path that runs tests and at least one pipeline smoke command.
- [x] Add docs for repo scope, supported commands, and artifact meanings.

Exit criteria:

- `pytest` passes
- synthetic train/eval/infer/demo all run from documented commands
- evaluation writes stable report files under `outputs/eval/`
- the adapter boundary is in place even if only synthetic data uses it initially

### P1: Complete Public V1

Focus: cross the line from `planner core` to `complete project`.

Target outcome:

- one public-format data path
- one deployment export path
- one lightweight closed-loop loop
- complete repo docs

Planned work:

- [x] Implement one public-format data adapter.
      Current implementation uses a normalized NPZ bridge format that is easy to document.
- [x] Add dataset normalization statistics and caching.
- [x] Add `scripts/train_planner.py` and `scripts/eval_planner.py` as user-facing entry points
      that work for both synthetic and adapter-backed data.
- [x] Add `scripts/export_onnx.py`.
- [x] Add ONNX vs PyTorch parity checking with fixed tolerance.
- [x] Add inference benchmark script for latency and throughput on CPU and GPU.
- [x] Add `planner/sim/` or `planner/rollout/` for lightweight closed-loop rollout.
- [x] Add rollout metrics: collision rate, route deviation, progress, comfort proxies, recovery rate.
      Current implementation provides a lightweight first pass, not a full simulator-grade metric suite.
- [x] Generate rollout visual artifacts: GIF, frame gallery, or video snippets.
      Current implementation provides a rollout trace plot rather than video assets.
- [x] Expand README with a full command matrix and project architecture diagram.
- [x] Add a release checklist for checkpoints, demo outputs, and docs.

Exit criteria:

- one documented non-synthetic data flow works end to end
- ONNX export succeeds and passes parity check
- closed-loop rollout demo produces metrics and visual artifacts
- README is sufficient for a third party to run the project

### P2: Model And Research Upgrade

Focus: improve quality after the project is already complete.

Target outcome:

- stronger model quality
- more credible planning selection
- ablation-backed improvements

Planned work:

- [ ] Replace heuristic candidate scoring with a learned scorer or value head.
- [ ] Replace heuristic candidate scoring with a learned scorer or value head.
      Groundwork is in place through an optional learned scorer head and hybrid scoring path.
- [ ] Upgrade the scene encoder toward explicit agent-map-route attention.
- [ ] Add ablation configs for encoder, noise mode, sampler, scorer, and route prior variants.
- [ ] Add scenario-specific failure analysis outputs.
- [ ] Add experiment tables to compare synthetic-only vs adapter-backed training.
- [ ] Consider reward modeling or offline RL fine-tuning only after the rollout layer is stable.

Exit criteria:

- one model upgrade shows a measurable gain over the current baseline
- ablation results are reproducible
- learned selection is better than heuristic selection on at least one reportable metric set

## Recommended File Evolution

The likely directory growth for `V1` should look like:

```text
planner/
  datasets/
    adapters/
  inference/
  metrics/
  rollout/
  export/
scripts/
  prepare_dataset.py
  train_planner.py
  eval_planner.py
  export_onnx.py
  benchmark_infer.py
docs/
  commands.md
  artifacts.md
  evaluation.md
```

Not every file must be created exactly as shown, but the capability boundaries should emerge.

## Execution Order

The recommended next implementation sequence is:

1. Build the dataset adapter interface and manifest generator.
2. Refactor evaluation into a stable report contract and artifact schema.
3. Unify train/eval/infer scripts around a single public CLI surface.
4. Add ONNX export and parity tests.
5. Add lightweight closed-loop rollout.
6. Only then start a larger model refactor.

This order keeps architecture pressure low and minimizes rework.

## Risks

The main risks are not model-related.

- Dataset scope explosion: trying to support too many formats too early.
- Evaluation churn: changing metric names and report structure after downstream scripts exist.
- Simulator overreach: building a heavy service instead of a small public rollout loop.
- Premature research work: spending time on new model blocks before data and evaluation are stable.

The mitigation is simple: finish `P0`, then finish `P1`, then optimize.

## Immediate Next Task

The best next coding task is:

- implement `planner/datasets/adapters/`
- add `scripts/prepare_dataset.py`
- define the first manifest format

That is the highest-leverage step toward a complete project.
