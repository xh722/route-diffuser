# RouteDiffuser Ablations

This document defines a small, stable ablation matrix for `RouteDiffuser`.

The goal is not to exhaustively search hyperparameters. The goal is to make a few comparisons easy
to rerun and easy to interpret.

## Current Scorer Config Set

Available scorer-related model configs:

- `configs/model/heuristic_only.yaml`
- `configs/model/learned_scorer_light.yaml`
- `configs/model/learned_scorer.yaml`
- `configs/model/learned_scorer_strong.yaml`

These correspond to:

| Config | `learned_scorer_weight` | Intended Use |
| --- | --- | --- |
| `heuristic_only.yaml` | `0.0` | baseline |
| `learned_scorer_light.yaml` | `0.05` | gentle hybrid influence |
| `learned_scorer.yaml` | `0.10` | default learned-scorer experiment |
| `learned_scorer_strong.yaml` | `0.25` | aggressive hybrid influence |

## Recommended Comparison Order

Run comparisons in this order:

1. `heuristic_only.yaml` vs `learned_scorer.yaml`
2. `heuristic_only.yaml` vs `learned_scorer_light.yaml`
3. `heuristic_only.yaml` vs `learned_scorer_strong.yaml`

That sequence tells you:

- whether the scorer helps at all
- whether small scorer influence is safer
- whether stronger scorer influence destabilizes selection

## Suggested Commands

Baseline comparison:

```bash
python scripts/compare_scorer.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/learned_scorer.yaml \
  --output outputs/eval/scorer_comparison_default.json \
  --device cpu
```

Light scorer:

```bash
python scripts/compare_scorer.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/learned_scorer_light.yaml \
  --output outputs/eval/scorer_comparison_light.json \
  --device cpu
```

Strong scorer:

```bash
python scripts/compare_scorer.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/learned_scorer_strong.yaml \
  --output outputs/eval/scorer_comparison_strong.json \
  --device cpu
```

Matrix shortcut:

```bash
python scripts/run_ablation_matrix.py \
  --matrix scorer \
  --output-dir outputs/ablations/scorer_matrix \
  --device cpu
```

## What To Watch

Primary metrics:

- `ade`
- `fde`
- `route_error`
- `collision_rate`
- `selected_score`

Interpretation guidance:

- lower `ade` and `fde` are better
- lower `route_error` is better
- lower `collision_rate` is better
- `selected_score` is only meaningful relative to another run using the same scoring mode

## Keep These Fixed

When comparing scorer configs, try to keep these constant:

- dataset config
- manifest
- checkpoint
- batch size
- candidate sample count
- evaluation seed

If you change more than one of those at once, the scorer comparison becomes noisy.

## Current Scope

These ablations only vary the scorer weight. They do not yet vary:

- scene encoder architecture
- candidate-set construction strategy
- scorer hidden dimension
- scorer candidate count
- scorer target temperature

Those can become the next ablation axes once the basic scorer comparison shows signal.

## Encoder Config Set

Available encoder-scale configs:

- `configs/model/base.yaml`
- `configs/model/encoder_small.yaml`
- `configs/model/encoder_wide.yaml`
- `configs/model/encoder_attention.yaml`

These correspond to:

| Config | `hidden_dim` | `time_dim` | `decoder_down_dims` | Fusion | Intended Use |
| --- | --- | --- | --- | --- | --- |
| `base.yaml` | `128` | `128` | `[128, 256]` | `concat_mlp` | default |
| `encoder_small.yaml` | `96` | `96` | `[96, 192]` | `concat_mlp` | lighter model / lower cost |
| `encoder_wide.yaml` | `192` | `192` | `[192, 384]` | `concat_mlp` | higher-capacity model |
| `encoder_attention.yaml` | `128` | `128` | `[128, 256]` | `token_attention` | explicit modality-token fusion |

## Recommended Encoder Comparison Order

Run comparisons in this order:

1. `base.yaml` vs `encoder_small.yaml`
2. `base.yaml` vs `encoder_wide.yaml`
3. `base.yaml` vs `encoder_attention.yaml`

That sequence tells you:

- whether the current model is overbuilt for the synthetic setting
- whether widening the encoder moves metrics enough to justify extra cost
- whether explicit attention fusion changes planning quality at the same hidden size

## Encoder Comparison Guidance

When comparing encoder configs, keep these fixed:

- scorer mode
- dataset config
- manifest
- checkpoint policy
- batch size
- diffusion step count
- evaluation seed

Do not mix encoder width changes with scorer-weight changes in the same comparison if you want
clean attribution.

Encoder matrix shortcut:

```bash
python scripts/run_ablation_matrix.py \
  --matrix encoder \
  --output-dir outputs/ablations/encoder_matrix \
  --device cpu
```
