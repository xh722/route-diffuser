# RouteDiffuser 命令手册

本文档说明 `RouteDiffuser` 的公开命令入口、常用参数和典型用法。

## 命令入口

可以用两种方式运行项目。

脚本入口：

- `python scripts/prepare_dataset.py`
- `python scripts/compute_dataset_stats.py`
- `python scripts/export_dataset_npz.py`
- `python scripts/train_planner.py`
- `python scripts/infer_planner.py`
- `python scripts/eval_planner.py`
- `python scripts/demo_planner.py`
- `python scripts/export_onnx.py`
- `python scripts/check_onnx_parity.py`
- `python scripts/benchmark_infer.py`
- `python scripts/rollout_planner.py`
- `python scripts/compare_scorer.py`
- `python scripts/run_ablation_matrix.py`
- `python scripts/analyze_failures.py`
- `python scripts/build_registry.py`

安装后命令别名：

- `route-diffuser-prepare`
- `route-diffuser-stats`
- `route-diffuser-export-npz`
- `route-diffuser-train`
- `route-diffuser-infer`
- `route-diffuser-eval`
- `route-diffuser-demo`
- `route-diffuser-export-onnx`
- `route-diffuser-check-onnx`
- `route-diffuser-benchmark`
- `route-diffuser-rollout`
- `route-diffuser-compare-scorer`
- `route-diffuser-ablations`
- `route-diffuser-failures`
- `route-diffuser-registry`

兼容旧入口：

- `python scripts/train_diffusion.py`
- `python scripts/infer_scene.py`
- `python scripts/eval_diffusion.py`
- `python scripts/demo_portfolio.py`

## 配置文件

默认配置：

- 数据：`configs/data/synthetic.yaml`
- NPZ 示例：`configs/data/npz_example.yaml`
- 模型：`configs/model/base.yaml`
- 训练：`configs/train/base.yaml`
- 推理：`configs/inference/base.yaml`

常用模型消融配置：

- `configs/model/heuristic_only.yaml`
- `configs/model/learned_scorer_light.yaml`
- `configs/model/learned_scorer.yaml`
- `configs/model/learned_scorer_strong.yaml`
- `configs/model/learned_scorer_drift.yaml`
- `configs/model/learned_scorer_mixed.yaml`
- `configs/model/learned_scorer_route_anchor.yaml`
- `configs/model/learned_scorer_reward.yaml`
- `configs/model/encoder_small.yaml`
- `configs/model/encoder_wide.yaml`
- `configs/model/encoder_attention.yaml`

原则：

- YAML 是主要控制面。
- CLI 参数只做少量运行时覆盖。
- 实验差异优先通过配置文件表达。

## 准备数据 Manifest

用途：生成固定数据切片，方便训练、评估和 demo 复现。

```bash
python scripts/prepare_dataset.py \
  --data-config configs/data/synthetic.yaml \
  --output outputs/manifests/route_diffuser_synthetic_train.json
```

常用参数：

- `--data-config`
- `--output`
- `--split`
- `--start-index`
- `--num-samples`

小样本示例：

```bash
python scripts/prepare_dataset.py \
  --output outputs/manifests/tiny_train.json \
  --split train \
  --start-index 0 \
  --num-samples 8
```

## 计算数据统计

用途：生成特征统计缓存，用于数据检查和归一化参考。

```bash
python scripts/compute_dataset_stats.py \
  --data-config configs/data/synthetic.yaml \
  --output outputs/stats/route_diffuser_synthetic_stats.json
```

常用参数：

- `--manifest-path`
- `--batch-size`
- `--max-scenes`
- `--output`

## 导出 NPZ Bridge 数据

用途：把 canonical samples 导出成公开 `.npz` 格式。

```bash
python scripts/export_dataset_npz.py \
  --data-config configs/data/synthetic.yaml \
  --output outputs/datasets/route_diffuser_synthetic.npz
```

导出后可在 `configs/data/npz_example.yaml` 中指定 `source_path`，再走同一套训练/评估流程。

## 训练 Planner

用途：训练扩散规划器并保存 checkpoint。

```bash
python scripts/train_planner.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --train-config configs/train/base.yaml
```

常用参数：

- `--manifest-path`
- `--output-dir`
- `--epochs`
- `--batch-size`
- `--device`

低负载示例：

```bash
python scripts/train_planner.py \
  --manifest-path outputs/manifests/tiny_train.json \
  --epochs 1 \
  --batch-size 2 \
  --device cpu
```

输出：

- `train_log.csv`
- `latest.pt`

## 推理 Planner

用途：采样未来轨迹并输出预测图。

```bash
python scripts/infer_planner.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --inference-config configs/inference/base.yaml
```

常用参数：

- `--checkpoint`
- `--manifest-path`
- `--output-dir`
- `--batch-size`
- `--num-samples`
- `--device`
- `--selection-mode`

输出：

- `predictions.pt`
- `prediction_plot.png`

## 评估 Planner

用途：生成结构化 open-loop evaluation report。

```bash
python scripts/eval_planner.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --inference-config configs/inference/base.yaml
```

常用参数：

- `--checkpoint`
- `--manifest-path`
- `--output-dir`
- `--batch-size`
- `--num-samples`
- `--device`
- `--selection-mode`

输出：

- `evaluation_report.json`
- `evaluation_report.md`

## 运行作品集 Demo

用途：训练一个小模型，生成适合简历和 GitHub 展示的完整产物。

```bash
python scripts/demo_planner.py \
  --epochs 20 \
  --device cpu
```

常用参数：

- `--model-config`
- `--data-config`
- `--train-config`
- `--inference-config`
- `--output-dir`
- `--epochs`
- `--num-samples`
- `--selection-mode`

输出目录：

- `outputs/portfolio_demo/`

## 导出 ONNX

用途：导出 denoiser core。

```bash
python scripts/export_onnx.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --output outputs/onnx/planner_denoiser.onnx
```

常用参数：

- `--checkpoint`
- `--manifest-path`
- `--metadata-output`
- `--batch-size`
- `--device`
- `--opset-version`
- `--static-batch`

注意：

- 导出的是 denoiser core，不是完整 DDPM sampling loop。

## 检查 ONNX Parity

用途：比较 ONNXRuntime 和 PyTorch wrapper 的数值一致性。

```bash
python scripts/check_onnx_parity.py \
  --onnx-path outputs/onnx/planner_denoiser.onnx \
  --data-config configs/data/synthetic.yaml
```

常用参数：

- `--onnx-path`
- `--data-config`
- `--model-config`
- `--checkpoint`
- `--tolerance`
- `--device`

## Benchmark

用途：统计推理延迟和吞吐。

```bash
python scripts/benchmark_infer.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --device cpu
```

常用参数：

- `--batch-size`
- `--warmup`
- `--iterations`
- `--num-samples`
- `--device`

## Closed-loop Rollout

用途：运行轻量 receding-horizon rollout。

```bash
python scripts/rollout_planner.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/base.yaml \
  --num-steps 8 \
  --device cpu
```

常用参数：

- `--checkpoint`
- `--manifest-path`
- `--output-dir`
- `--num-steps`
- `--num-samples`
- `--selection-mode`

输出：

- `rollout_trace.pt`
- `rollout_summary.json`
- `rollout_summary.md`
- `rollout_plot.png`

## Scorer 对比

用途：同一批样本下比较 heuristic 和 hybrid scorer。

```bash
python scripts/compare_scorer.py \
  --data-config configs/data/synthetic.yaml \
  --model-config configs/model/learned_scorer_route_anchor.yaml \
  --output outputs/eval/scorer_comparison_route_anchor.json \
  --device cpu
```

重点观察：

- ADE/FDE delta
- route error delta
- box collision rate delta
- heuristic regret
- learned preference regret

## 消融实验矩阵

Scorer matrix：

```bash
python scripts/run_ablation_matrix.py \
  --matrix scorer \
  --output-dir outputs/ablations/scorer_matrix \
  --device cpu
```

Encoder matrix：

```bash
python scripts/run_ablation_matrix.py \
  --matrix encoder \
  --output-dir outputs/ablations/encoder_matrix \
  --device cpu
```

自定义模型列表：

```bash
python scripts/run_ablation_matrix.py \
  --matrix scorer \
  --model-configs \
    configs/model/heuristic_only.yaml \
    configs/model/learned_scorer.yaml \
    configs/model/learned_scorer_route_anchor.yaml \
  --device cpu
```

## Failure Analysis

用途：按指标排序失败样例，定位高风险场景。

```bash
python scripts/analyze_failures.py \
  --ranking-metric fde \
  --top-k 5 \
  --device cpu
```

常用 ranking metric：

- `ade`
- `fde`
- `route_error`
- `collision`
- `comfort_violation`

## 构建 Registry

用途：汇总多个评估、失败分析和消融结果。

```bash
python scripts/build_registry.py \
  --input-dir outputs \
  --output outputs/registry/registry.json
```

输出：

- `registry.json`
- `leaderboard.md`

## 推荐验证命令

语法检查：

```bash
python -m compileall planner scripts tests
```

单元测试：

```bash
pytest -q
```

最小 smoke：

```bash
python scripts/prepare_dataset.py --num-samples 4 --output outputs/manifests/tiny.json
python scripts/train_planner.py --manifest-path outputs/manifests/tiny.json --epochs 1 --batch-size 2 --device cpu
python scripts/eval_planner.py --manifest-path outputs/manifests/tiny.json --batch-size 2 --num-samples 1 --device cpu
```
