# RouteDiffuser 消融实验说明

本文档说明 `RouteDiffuser` 当前支持的 scorer、candidate strategy 和 encoder 消融实验。

## Scorer 配置集合

可用配置：

- `configs/model/heuristic_only.yaml`
- `configs/model/learned_scorer_light.yaml`
- `configs/model/learned_scorer.yaml`
- `configs/model/learned_scorer_strong.yaml`
- `configs/model/learned_scorer_drift.yaml`
- `configs/model/learned_scorer_mixed.yaml`
- `configs/model/learned_scorer_route_anchor.yaml`
- `configs/model/learned_scorer_reward.yaml`

配置含义：

| 配置 | `learned_scorer_weight` | Candidate Strategy | Target Mode | 用途 |
| --- | --- | --- | --- | --- |
| `heuristic_only.yaml` | `0.0` | `gt_prior_noise` | `ade` | 纯启发式 baseline |
| `learned_scorer_light.yaml` | `0.05` | `gt_prior_noise` | `ade` | 轻量 learned scorer 影响 |
| `learned_scorer.yaml` | `0.10` | `gt_prior_noise` | `ade` | 默认 learned scorer 实验 |
| `learned_scorer_strong.yaml` | `0.25` | `gt_prior_noise` | `ade` | 更强 learned scorer 影响 |
| `learned_scorer_drift.yaml` | `0.10` | `gt_prior_drift` | `ade` | 漂移候选实验 |
| `learned_scorer_mixed.yaml` | `0.10` | `mixed` | `ade` | noise + drift 混合候选 |
| `learned_scorer_route_anchor.yaml` | `0.10` | `route_anchor` | `ade` | lane-intention 风格 route anchor |
| `learned_scorer_reward.yaml` | `0.10` | `gt_prior_noise` | `reward` | reward-aware scorer supervision |

## 推荐比较顺序

1. `heuristic_only.yaml` vs `learned_scorer.yaml`
2. `heuristic_only.yaml` vs `learned_scorer_light.yaml`
3. `heuristic_only.yaml` vs `learned_scorer_strong.yaml`
4. `learned_scorer.yaml` vs `learned_scorer_drift.yaml`
5. `learned_scorer.yaml` vs `learned_scorer_mixed.yaml`
6. `learned_scorer.yaml` vs `learned_scorer_route_anchor.yaml`
7. `learned_scorer.yaml` vs `learned_scorer_reward.yaml`

这组对比可以回答：

- learned scorer 是否带来收益。
- scorer 权重变大后是否会破坏启发式安全选择。
- 结构化候选是否比纯 noise 更适合训练 scorer。
- route-anchor 是否能提供更接近真实规划意图的候选。
- reward-aware target 与 ADE target 的行为差异。

## 常用命令

默认 scorer 对比：

```bash
python scripts/compare_scorer.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/learned_scorer.yaml \
  --output outputs/eval/scorer_comparison_default.json \
  --device cpu
```

route-anchor scorer 对比：

```bash
python scripts/compare_scorer.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/learned_scorer_route_anchor.yaml \
  --output outputs/eval/scorer_comparison_route_anchor.json \
  --device cpu
```

scorer matrix：

```bash
python scripts/run_ablation_matrix.py \
  --matrix scorer \
  --output-dir outputs/ablations/scorer_matrix \
  --device cpu
```

自定义配置列表：

```bash
python scripts/run_ablation_matrix.py \
  --matrix scorer \
  --model-configs \
    configs/model/heuristic_only.yaml \
    configs/model/learned_scorer.yaml \
    configs/model/learned_scorer_route_anchor.yaml \
  --output-dir outputs/ablations/custom_scorer_matrix \
  --device cpu
```

## Candidate Strategy

当前支持：

- `gt_prior_noise`：ground truth / route prior 加噪声扰动。
- `gt_prior_drift`：构造纵向和横向 drift 候选。
- `mixed`：同时使用 drift 和 noise。
- `route_anchor`：基于 route prior 构造居中、左偏、右偏、快速、慢速候选。

`route_anchor` 的意义：

- 更接近真实规划系统里的 intention / anchor 思路。
- 能让 scorer 看到结构化候选，而不是只学习随机扰动。
- 适合作为 lane-level decision 的公开简化版。

## 重点观察指标

主要指标：

- `ade`
- `fde`
- `route_error`
- `box_collision_rate`
- `point_collision_rate`
- `comfort_violation_rate`
- `selected_score`
- `heuristic_regret`
- `matches_heuristic_best`
- `matches_learned_best`
- `learned_preference_regret`

解释：

- `ade` / `fde` 越低越好。
- `route_error` 越低说明路线一致性越好。
- `box_collision_rate` 是主要安全指标，越低越好。
- `point_collision_rate` 保留为对照指标。
- `selected_score` 只适合在同一 scoring mode 下横向比较。
- `heuristic_regret` 越低，说明选择越接近启发式最优候选。
- `matches_heuristic_best` 越高，说明 scorer 与启发式 baseline 越一致。
- `matches_learned_best` 和 `learned_preference_regret` 用于诊断 hybrid scorer 是否跟随 learned preference。

## Encoder 配置集合

可用配置：

- `configs/model/base.yaml`
- `configs/model/encoder_small.yaml`
- `configs/model/encoder_wide.yaml`
- `configs/model/encoder_attention.yaml`

配置含义：

| 配置 | `hidden_dim` | `time_dim` | `decoder_down_dims` | Fusion | 用途 |
| --- | --- | --- | --- | --- | --- |
| `base.yaml` | `128` | `128` | `[128, 256]` | `concat_mlp` | 默认模型 |
| `encoder_small.yaml` | `96` | `96` | `[96, 192]` | `concat_mlp` | 更轻量 |
| `encoder_wide.yaml` | `192` | `192` | `[192, 384]` | `concat_mlp` | 更高容量 |
| `encoder_attention.yaml` | `128` | `128` | `[128, 256]` | `token_attention` | 显式 modality token fusion |

推荐比较：

1. `base.yaml` vs `encoder_small.yaml`
2. `base.yaml` vs `encoder_wide.yaml`
3. `base.yaml` vs `encoder_attention.yaml`

命令：

```bash
python scripts/run_ablation_matrix.py \
  --matrix encoder \
  --output-dir outputs/ablations/encoder_matrix \
  --device cpu
```

## 保持固定的变量

比较实验时尽量固定：

- dataset config
- manifest
- checkpoint policy
- batch size
- candidate sample count
- random seed
- selection mode

如果一次改变多个变量，scorer 和模型结构的影响会很难解释。

## 输出产物

`run_ablation_matrix.py` 会输出：

- `*_ablation_matrix.json`
- `*_ablation_matrix.md`

这些产物会记录：

- config summary
- selection strategy
- overall metrics
- candidate-set metrics
- 相对 baseline 的 delta

## 面试讲解建议

可以把消融实验讲成三层：

1. **Baseline**：纯 heuristic scorer 是否能提供稳定选择。
2. **Scorer upgrade**：learned scorer 是否改变选择结果，是否引入风险。
3. **Candidate strategy**：noise、drift、route-anchor 哪种候选更适合作为 scorer supervision。
