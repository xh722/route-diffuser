# RouteDiffuser 架构说明

本文档说明 `RouteDiffuser` 的模块边界、数据流和关键设计取舍。

## 总体数据流

```text
数据适配器 / synthetic generator
  -> CanonicalSceneBatch
  -> SceneEncoder
  -> route prior
  -> DiffusionDecoder
  -> sampled trajectory candidates
  -> trajectory scorer
  -> metrics / reports / visualization / rollout
```

核心原则：

- 所有模型和评估逻辑都围绕统一的 `CanonicalSceneBatch`。
- 模型生成和候选选择分离，便于替换 scorer。
- open-loop evaluation、rollout 和 demo 共用同一套 metrics。
- 公开仓库只保留可运行、可解释、无私有依赖的 planner core。

## `planner/datasets/`

职责：

- 定义 planner-facing 的 canonical scene schema。
- 提供 synthetic 数据集。
- 提供 adapter-backed dataset boundary。
- 计算数据统计缓存。

关键文件：

- `schema.py`：`CanonicalSceneBatch`，包含 ego、neighbors、lanes、route、future trajectory 和 mask。
- `synthetic.py`：兼容旧接口的 synthetic dataset wrapper。
- `factory.py`：根据配置构造 dataset / dataloader。
- `statistics.py`：计算 feature 统计信息。
- `adapters/`：synthetic 和 NPZ bridge adapter。

设计意义：

- 训练、推理、评估、导出不需要关心数据来源。
- 后续接公开真实数据时，只需要实现 adapter，不需要重写 planner。

## `planner/preprocess/`

职责：

- 提供坐标、角度和路线先验相关预处理。

关键文件：

- `angles.py`：heading 与 cos/sin 表示转换。
- `coordinates.py`：局部坐标和几何工具。
- `route_prior.py`：根据 route polyline 构造未来轨迹 prior。
- `normalization.py`：归一化相关工具。

设计意义：

- route prior 是模型生成的显式条件。
- heading 使用 cos/sin 可以避免角度跳变。

## `planner/models/`

职责：

- 定义 route-conditioned diffusion planner。
- 编码场景上下文。
- 解码轨迹残差。
- 对候选轨迹进行 learned scoring。

关键文件：

- `diffusion_planner.py`：主模型、训练 loss、采样、scorer candidate set。
- `scene_encoder.py`：concat MLP 和 token attention 两种 scene fusion。
- `diffusion_decoder.py`：条件 1D U-Net denoiser。
- `trajectory_scorer.py`：候选轨迹 scorer head。

当前支持的模型实验轴：

- `hidden_dim` / `time_dim` / `decoder_down_dims`
- `scene_fusion_mode`: `concat_mlp` 或 `token_attention`
- `diffusion_noise_mode`: 标准噪声或 pyramid 噪声
- `scorer_candidate_strategy`: `gt_prior_noise`、`gt_prior_drift`、`mixed`、`route_anchor`
- `scorer_target_mode`: `ade` 或 `reward`

## `planner/diffusion/`

职责：

- 提供 diffusion schedule、噪声采样和 DDPM step。

关键文件：

- `schedule.py`：linear beta schedule。
- `noise.py`：标准噪声和 pyramid noise。
- `utils.py`：`q_sample` 和 `ddpm_step`。

设计意义：

- diffusion 工具与 planner 模型解耦，便于后续替换 sampler。

## `planner/inference/`

职责：

- 执行候选轨迹选择。
- 将首帧锚定到当前 ego 状态。
- 输出 scorer 诊断指标。

关键文件：

- `anchoring.py`：保证预测轨迹首帧等于当前 ego 状态。
- `scoring.py`：route、clearance、comfort、learned scorer 的 hybrid scoring。

scoring 输出包括：

- `scores`
- `heuristic_scores`
- `selected_indices`
- `selected_trajectories`
- `heuristic_regret`
- `learned_preference_regret`
- `matches_heuristic_best`
- learned scorer 相关归一化诊断

设计意义：

- 生成和选择分离，真实规划系统中这通常也是两个不同职责。
- scorer 诊断能帮助判断 learned scorer 是否真的改善选择。

## `planner/metrics/`

职责：

- 提供 open-loop、candidate-set、comfort、clearance 和 collision 指标。

关键文件：

- `trajectory.py`：ADE/FDE、route error、progress、comfort、clearance、candidate oracle metrics。
- `collision.py`：oriented-box collision。

安全指标：

- `point_collision_rate`：基于点距离阈值。
- `box_collision_rate`：基于车辆矩形 footprint，当前主要安全指标。
- `collision_rate`：兼容字段，当前等价于 box collision。

设计意义：

- box collision 比点距离更接近真实车辆占用空间。
- open-loop evaluation 和 closed-loop rollout 使用一致的安全定义。

## `planner/trainers/`

职责：

- 提供训练循环和评估入口。

关键文件：

- `diffusion_trainer.py`：optimizer、one-epoch training、detailed evaluation。

设计意义：

- 训练、评估和报告生成共用稳定路径。
- 评估报告可直接被 demo、ablation 和 registry 复用。

## `planner/reports/`

职责：

- 定义结构化报告和实验结果组织方式。

关键文件：

- `evaluation.py`：evaluation report schema、JSON/Markdown 序列化。
- `failure_analysis.py`：失败样例排序和报告。
- `registry.py`：实验 registry / leaderboard。

设计意义：

- 项目不是只打印指标，而是产出可比较、可归档、可展示的 report。

## `planner/export/`

职责：

- 导出和验证部署相关产物。

关键文件：

- `onnx.py`：导出 denoiser core。
- `parity.py`：ONNX 和 PyTorch 数值一致性检查。
- `benchmark.py`：延迟和吞吐统计。

注意：

- 当前 ONNX 导出的是 denoiser core，不是完整 DDPM sampling loop。

## `planner/rollout/`

职责：

- 提供轻量 closed-loop replanning。

关键文件：

- `simulator.py`：单场景 rollout、世界/局部坐标转换、summary。

rollout 输出：

- executed world states
- reference world states
- selected candidate indices
- selected scores
- box collision flags
- point collision flags
- route error / closed-loop ADE/FDE

设计意义：

- 展示 planner 在 receding-horizon setting 下的闭环行为。
- 保持轻量，不依赖私有仿真服务。

## `planner/visualization/`

职责：

- 生成轨迹对比图、候选轨迹图、场景画廊和 rollout plot。

设计意义：

- 自动驾驶规划项目必须能可视化，否则很难解释模型行为。

## `scripts/` 与 `planner/cli/`

职责：

- `planner/cli/` 存放真正的 CLI 实现。
- `scripts/` 提供公开脚本入口和旧命令兼容 wrapper。

设计意义：

- 保持命令入口清晰。
- 兼容旧脚本名，减少使用成本。

## 配置系统

配置目录：

- `configs/data/`
- `configs/model/`
- `configs/train/`
- `configs/inference/`

设计原则：

- YAML 是主要控制面。
- CLI flag 只做小范围运行时覆盖。
- 消融实验通过配置文件表达，而不是硬编码在脚本里。

## 当前边界

包含：

- planner core
- diffusion trajectory generation
- candidate scoring
- metrics/reporting
- visualization
- ONNX export
- lightweight rollout

不包含：

- 私有数据格式
- 私有仿真服务
- production deployment plugin
- 完整 RL training stack

这个边界保证项目可以公开、可复现、可解释。
