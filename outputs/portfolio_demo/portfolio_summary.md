# RouteDiffuser

> 基于路线条件约束的自动驾驶轨迹规划扩散模型项目

这是一个 planner-core 自动驾驶项目，展示统一场景建模、route-conditioned diffusion inference、多候选轨迹选择和端到端调试产物，不依赖私有日志或公司内部仿真服务。

## 项目亮点

- 统一的场景 schema，覆盖 ego、邻车、车道线、route polyline 和 mask。
- 基于 route prior residual 的条件扩散规划模型。
- 条件 1D U-Net 轨迹去噪器和迭代采样器。
- 多候选轨迹选择，不默认取第一个 sample。
- 结构化 synthetic 场景，覆盖直行、左变道、右变道和缓弯。
- 评估报告、候选轨迹图、场景画廊和 portfolio summary 产物齐全。

## 场景覆盖

- `keep_lane`：直线车道中心保持。
- `lane_change_left`：向左变道并保持前向进度。
- `lane_change_right`：向右变道并保持前向进度。
- `gentle_curve`：缓弯道路上的路线跟随。

## Open-loop 指标

- ADE：1.305
- FDE：1.724
- Route Error：1.061
- Progress：37.403
- Min Clearance：2.263
- Collision Rate：0.375
- Comfort Violation Rate：1.0

## Candidate-set 指标

- Oracle ADE：1.201
- Oracle FDE：1.036
- Oracle Route Error：0.942
- Final-state Diversity：1.908

## 场景拆分

- `gentle_curve`：ADE 1.52，FDE 2.815，Route Error 1.045
- `keep_lane`：ADE 1.211，FDE 1.306，Route Error 1.085
- `lane_change_left`：ADE 1.247，FDE 1.36，Route Error 1.076
- `lane_change_right`：ADE 1.243，FDE 1.415，Route Error 1.039

## 选择策略

- Strategy：`heuristic_route_clearance_comfort_scoring`
- Candidate Samples：3
- Mean Selected Index：1.052
- Mean Selected Score：27.579

## 简历表述

- 基于 PyTorch 实现 route-conditioned 自动驾驶轨迹规划器，核心为条件扩散模型和 1D U-Net 去噪解码器。
- 实现 route-prior residual diffusion、多候选轨迹生成和启发式/学习式混合评分选择。
- 构建场景级评估、候选轨迹可视化和作品集产物生成流程。

## 产物

- Checkpoint：`outputs/portfolio_demo/demo_checkpoint.pt`
- Predictions：`outputs/portfolio_demo/predictions.pt`
- Plot：`outputs/portfolio_demo/prediction_plot.png`
- Candidate Plot：`outputs/portfolio_demo/candidate_trajectories.png`
- Scenario Gallery：`outputs/portfolio_demo/scenario_gallery.png`
- JSON Summary：`outputs/portfolio_demo/portfolio_summary.json`
- Markdown Summary：`outputs/portfolio_demo/portfolio_summary.md`
