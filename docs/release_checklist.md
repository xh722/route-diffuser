# RouteDiffuser Release Checklist

Use this checklist before publishing a new checkpoint, demo refresh, or milestone update.

## Scope

Confirm what kind of release this is:

- [ ] docs-only update
- [ ] code + docs update
- [ ] artifact refresh
- [ ] model checkpoint refresh
- [ ] milestone / portfolio refresh

## Working Tree

- [ ] `git status` is understood and intentional
- [ ] no accidental outputs or scratch files are included
- [ ] large binary artifacts are included only when they are part of the intended release

## Commands And Entrypoints

- [ ] `docs/commands.md` still matches the public scripts
- [ ] public scripts listed in the README still exist
- [ ] any new CLI alias added in `pyproject.toml` is documented
- [ ] compatibility wrappers still point at the correct CLI modules

## Data Layer

- [ ] manifest generation still works for the documented path
- [ ] adapter config examples are still valid
- [ ] any new public-format adapter is documented with required keys and shapes

## Training Artifacts

- [ ] checkpoint naming is intentional
- [ ] training log format has not changed unexpectedly
- [ ] config files used to produce shared artifacts are committed

## Evaluation Layer

- [ ] evaluation report schema changes are intentional
- [ ] JSON and Markdown report outputs are still documented
- [ ] metric naming changes, if any, are reflected in docs and downstream summary files

## Deployment Layer

- [ ] ONNX export command is still documented
- [ ] parity command is still documented
- [ ] benchmark command is still documented
- [ ] export artifacts use stable filenames when possible

## Closed-Loop Rollout

- [ ] rollout command is still documented
- [ ] rollout artifact names are still documented
- [ ] rollout metrics remain interpretable and are not silently renamed

## Demo And Portfolio

- [ ] `outputs/portfolio_demo/` reflects the intended project state
- [ ] `portfolio_summary.json` and `.md` are consistent with current features
- [ ] hero images in the README still exist and render correctly
- [ ] scenario gallery and candidate trajectory plots still correspond to the current pipeline

## Documentation

- [ ] `README.md` reflects the current public entry points
- [ ] `README.zh-CN.md` is updated if the English README changed materially
- [ ] `docs/artifacts.md` reflects current output directories and filenames
- [ ] `docs/architecture.md` still matches the code structure
- [ ] `ROADMAP.md` reflects current progress honestly

## Validation

Pick the lightest validation that is appropriate for the release:

- [ ] syntax-only check
- [ ] targeted unit tests
- [ ] targeted CLI smoke commands
- [ ] artifact refresh commands

Record what was actually run in the commit message, PR description, or release notes.

## Final Review

- [ ] commit messages are coherent
- [ ] generated artifacts are worth versioning
- [ ] the release can be understood by someone opening the repo for the first time
