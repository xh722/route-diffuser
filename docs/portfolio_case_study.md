# RouteDiffuser 简历项目案例说明

这份文档用于把 `RouteDiffuser` 组织成一个适合社招简历、GitHub 项目页和面试讲解的案例。

## 项目一句话

`RouteDiffuser` 是一个基于路线条件约束的自动驾驶轨迹规划项目，使用 PyTorch 实现条件扩散模型、多候选轨迹生成、混合评分选择、结构化评估报告和轻量闭环 rollout。

## 项目背景

真实自动驾驶规划项目通常无法公开公司日志、仿真平台和部署链路，所以公开项目很容易停留在“模型片段”。这个仓库刻意选择一个可公开、可复现、可解释的边界：只展示 planner core，并把数据接口、模型、评估、可视化和部署导出补齐。

项目关注的问题是：

- 如何把 ego、邻车、车道线和 route 组织成稳定的 planner 输入。
- 如何基于 route prior 生成多条未来轨迹。
- 如何在多条候选轨迹中进行安全、舒适、路线一致的选择。
- 如何用结构化报告说明模型表现，而不是只给一个平均 ADE。

## 技术架构

```text
数据生成器 / 公开数据适配器
  -> canonical scene tensors
  -> scene encoder
  -> route prior
  -> diffusion residual decoder
  -> candidate trajectory bundle
  -> heuristic / learned hybrid scorer
  -> evaluation report / visualization / rollout
```

核心模块：

- `planner/datasets/`：统一场景 schema、synthetic 数据、NPZ bridge、manifest。
- `planner/models/`：条件扩散规划器、scene encoder、1D U-Net decoder、trajectory scorer。
- `planner/inference/`：候选轨迹打分、首帧锚定、hybrid selection。
- `planner/metrics/`：ADE/FDE、route error、comfort、clearance、oriented-box collision。
- `planner/reports/`：评估报告、失败分析和实验 registry。
- `planner/export/`：ONNX 导出、parity 和 benchmark。
- `planner/rollout/`：轻量闭环 rollout。

## 关键工程决策

### 1. Route-prior residual diffusion

模型不是直接生成绝对轨迹，而是先根据 route 构造路线先验，再让 diffusion decoder 预测残差。这样可以把导航意图显式注入生成过程，降低模型只靠数据学习 route adherence 的压力。

### 2. Canonical scene schema

所有训练、推理、评估和导出都围绕同一个 `CanonicalSceneBatch` 展开。这样 synthetic 数据、NPZ 数据和未来真实数据适配器可以共用一套 planner pipeline。

### 3. 多候选轨迹选择

扩散模型天然适合采样多条候选轨迹。项目没有默认取第一个 sample，而是引入 route、clearance、comfort、box collision 和 learned scorer 信号进行排序。

### 4. Route-anchor candidate strategy

参考真实规划系统中的 intention / anchor 思路，公开版实现了 `route_anchor` 候选集：路线居中、左偏、右偏、快速、慢速。这让 scorer 训练不只依赖随机 noise，也能看到结构化决策候选。

### 5. Oriented-box collision

点距离阈值无法准确表达车辆 footprint。项目新增 oriented-box collision，并同时接入 open-loop evaluation 和 closed-loop rollout，让安全指标更接近真实规划评估。

### 6. 评估可解释性

除了 ADE/FDE，报告还包含：

- route error
- progress
- min clearance
- point collision rate
- box collision rate
- comfort violation rate
- oracle candidate metrics
- scenario breakdown
- heuristic regret
- learned preference regret
- selected-candidate agreement

这些指标能解释“为什么选这条轨迹”，而不只是说明平均误差。

## 面试展示路径

建议面试时按这个顺序展示：

1. `README.md`：快速说明项目范围和产物。
2. `outputs/portfolio_demo/scenario_gallery.png`：展示场景覆盖。
3. `outputs/portfolio_demo/candidate_trajectories.png`：展示多候选规划。
4. `docs/architecture.md`：讲清模块边界。
5. `docs/ablations.md`：展示 scorer / encoder 的实验轴。
6. `docs/artifacts.md`：说明训练、评估、导出、rollout 产物。

## 简历写法建议

可以按下面风格写：

- 基于 PyTorch 实现 route-conditioned 自动驾驶轨迹规划器，采用条件扩散模型和 1D U-Net denoising decoder 生成未来轨迹。
- 设计 canonical planner scene schema，统一 ego、neighbors、lanes、route polylines、masks 和 future trajectory targets。
- 实现 route-prior residual diffusion、多候选轨迹采样、hybrid heuristic/learned scoring 和 route-anchor candidate supervision。
- 构建规划评估体系，覆盖 ADE/FDE、route consistency、comfort proxy、oriented-box collision、candidate oracle metrics、scenario breakdown 和 failure analysis。
- 补齐数据 manifest、NPZ bridge、ONNX export/parity、benchmark、closed-loop rollout、ablation matrix 和 portfolio artifacts。

## 面试可讲点

### 为什么用 diffusion？

扩散模型可以自然采样多条合理未来轨迹，适合表达多模态驾驶决策，也方便在 inference 阶段做 candidate scoring。

### 为什么要 route prior？

自动驾驶规划不是无条件生成轨迹。route prior 可以把导航意图直接放进模型，使生成结果更贴近规划任务。

### 为什么需要 scorer？

真实规划系统通常不会只生成一条轨迹，而是生成候选集后做选择。scorer 层让项目更接近真实规划架构，也让安全、舒适、路线一致性可以显式进入选择逻辑。

### 为什么要 box collision？

车辆有长宽和朝向，只用点距离会误判或漏判。oriented-box collision 更接近车辆 footprint overlap，因此更适合作为规划安全指标。

### 为什么不做完整仿真器？

公开项目的目标是展示 planner core。完整仿真器需要大量私有基础设施，反而会降低可复现性。轻量 rollout 能展示闭环思想，同时保持仓库可运行。

## 当前限制

- synthetic 场景利于复现，但真实公开数据会更有说服力。
- ONNX 导出的是 denoiser core，不是完整 DDPM sampling loop。
- rollout 是轻量闭环，不等价于生产级仿真平台。
- learned scorer 仍是实验轴，需要更多数据和 calibration 才能作为强选择器。

## 后续高价值优化

- 增加一个公开数据样例，完整走通 NPZ bridge。
- 增加 scorer calibration report。
- 增加 collision / route deviation / comfort violation failure gallery。
- 增加 model card 和 data card。
- 将 scene encoder 升级为 ego-query cross attention。
