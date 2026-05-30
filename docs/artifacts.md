# RouteDiffuser 产物说明

本文档说明各类命令会生成哪些文件，以及这些文件适合怎么使用。

## 目录约定

默认输出目录：

- `outputs/manifests/`：数据 manifest。
- `outputs/stats/`：数据统计缓存。
- `outputs/train/`：训练日志和 checkpoint。
- `outputs/infer/`：推理结果和预测图。
- `outputs/eval/`：评估报告、scorer 对比、failure analysis。
- `outputs/onnx/`：ONNX 模型和 metadata。
- `outputs/benchmarks/`：benchmark 报告。
- `outputs/rollout/`：闭环 rollout 产物。
- `outputs/ablations/`：消融实验矩阵。
- `outputs/portfolio_demo/`：简历展示 demo 产物。

## Manifest 产物

生成命令：

```bash
python scripts/prepare_dataset.py \
  --output outputs/manifests/route_diffuser_synthetic_train.json
```

主要文件：

- `*.json`

用途：

- 固定训练、评估或 demo 使用的数据切片。
- 记录 split、sample index、scenario name、entry id。
- 让实验不依赖脚本里的临时切样本逻辑。

## 数据统计产物

生成命令：

```bash
python scripts/compute_dataset_stats.py \
  --output outputs/stats/route_diffuser_synthetic_stats.json
```

主要文件：

- `*_stats.json`

用途：

- 缓存 ego、neighbor、lane、route、future trajectory 等张量的统计信息。
- 用于数据检查、归一化参考和实验记录。

## NPZ Bridge 产物

生成命令：

```bash
python scripts/export_dataset_npz.py \
  --output outputs/datasets/route_diffuser_synthetic.npz
```

主要文件：

- `*.npz`

用途：

- 把 canonical planning samples 导出为公开、简单、可复用的数据格式。
- 可被 `configs/data/npz_example.yaml` 加载。

NPZ 关键字段：

- `ego_current_state`
- `neighbor_history`
- `neighbor_history_mask`
- `lane_polylines`
- `lane_polylines_mask`
- `route_lanes`
- `route_lanes_mask`
- `future_ego_trajectory`
- `future_ego_mask`

## 训练产物

生成命令：

```bash
python scripts/train_planner.py
```

主要文件：

- `latest.pt`
- `train_log.csv`

用途：

- `latest.pt` 保存模型参数和配置。
- `train_log.csv` 保存每个 epoch 的 loss、diffusion loss、scorer loss 和 scorer accuracy。

## 推理产物

生成命令：

```bash
python scripts/infer_planner.py
```

主要文件：

- `predictions.pt`
- `prediction_plot.png`

用途：

- `predictions.pt` 保存采样的候选轨迹张量。
- `prediction_plot.png` 展示预测轨迹、目标轨迹和 route。

## 评估报告

生成命令：

```bash
python scripts/eval_planner.py
```

主要文件：

- `evaluation_report.json`
- `evaluation_report.md`

报告内容：

- dataset 信息
- selection strategy
- overall metrics
- candidate-set metrics
- scenario metrics
- artifacts
- metadata

常见指标：

- `ade`
- `fde`
- `route_error`
- `progress`
- `min_clearance`
- `box_collision_rate`
- `point_collision_rate`
- `comfort_violation_rate`
- `oracle_ade`
- `oracle_fde`
- `candidate_final_diversity`

## Scorer 对比产物

生成命令：

```bash
python scripts/compare_scorer.py
```

主要文件：

- `scorer_comparison.json`

用途：

- 对比 heuristic 和 hybrid scorer。
- 输出 overall/candidate metrics 的差值。
- 观察 learned scorer 是否改善选择，或是否引入风险。

## 消融实验产物

生成命令：

```bash
python scripts/run_ablation_matrix.py --matrix scorer
python scripts/run_ablation_matrix.py --matrix encoder
```

主要文件：

- `scorer_ablation_matrix.json`
- `scorer_ablation_matrix.md`
- `encoder_ablation_matrix.json`
- `encoder_ablation_matrix.md`

用途：

- 对比不同 scorer 权重、candidate strategy、target mode 和 encoder 配置。
- 记录相对 baseline 的 delta。
- 适合放入实验记录或面试讲解材料。

## Failure Analysis 产物

生成命令：

```bash
python scripts/analyze_failures.py
```

主要文件：

- `failure_analysis.json`
- `failure_analysis.md`

用途：

- 按指定指标排序失败样例。
- 输出 overall top failures 和 scenario-level top failures。
- 帮助定位 route deviation、collision、comfort violation 等问题。

## Registry 产物

生成命令：

```bash
python scripts/build_registry.py
```

主要文件：

- `registry.json`
- `leaderboard.md`

用途：

- 汇总 evaluation、failure analysis、ablation 等结果。
- 给多个实验提供统一索引。

## ONNX 产物

生成命令：

```bash
python scripts/export_onnx.py \
  --output outputs/onnx/planner_denoiser.onnx
```

主要文件：

- `planner_denoiser.onnx`
- `planner_denoiser.metadata.json`

用途：

- 导出 denoiser core。
- 记录输入输出 shape、opset、配置和导出参数。

注意：

- 当前导出的是 denoiser core，不是完整 iterative DDPM sampler。

## ONNX Parity 产物

生成命令：

```bash
python scripts/check_onnx_parity.py \
  --onnx-path outputs/onnx/planner_denoiser.onnx
```

主要文件：

- `onnx_parity_report.json`

用途：

- 对比 ONNXRuntime 和 PyTorch wrapper 输出。
- 检查 max error、mean error 和是否通过 tolerance。

## Benchmark 产物

生成命令：

```bash
python scripts/benchmark_infer.py
```

主要文件：

- `benchmark_report.json`

用途：

- 记录 latency、throughput、batch size、device 等信息。
- 支持 CPU/GPU 运行环境对比。

## Rollout 产物

生成命令：

```bash
python scripts/rollout_planner.py
```

主要文件：

- `rollout_trace.pt`
- `rollout_summary.json`
- `rollout_summary.md`
- `rollout_plot.png`

用途：

- `rollout_trace.pt` 保存 executed states、reference states、route、selected indices、selected scores、box collision flags 和 point collision flags。
- `rollout_summary.json/.md` 保存 closed-loop ADE/FDE、route error、collision rate 等指标。
- `rollout_plot.png` 展示执行轨迹、参考轨迹和 route。

## 作品集 Demo 产物

生成命令：

```bash
python scripts/demo_planner.py
```

主要文件：

- `demo_checkpoint.pt`
- `predictions.pt`
- `prediction_plot.png`
- `candidate_trajectories.png`
- `scenario_gallery.png`
- `evaluation_report.json`
- `evaluation_report.md`
- `portfolio_summary.json`
- `portfolio_summary.md`

用途：

- 作为 GitHub 首页、简历项目链接和面试讲解材料。
- 展示训练、推理、评估、候选选择和可视化的完整闭环。

## 哪些产物适合提交

通常适合提交：

- 小型 demo 图片。
- 小型 summary JSON / Markdown。
- 示例配置和文档。

谨慎提交：

- 大型 checkpoint。
- 大型 `.pt` 或 `.npz` 文件。
- 大量临时实验输出。

提交前请参考 [release_checklist.md](release_checklist.md)。
