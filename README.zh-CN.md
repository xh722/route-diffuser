# RouteDiffuser

> 一个基于路线条件约束的自动驾驶轨迹规划扩散模型项目。

[English README](README.md)

`RouteDiffuser` 是 `nn_planner` 这套代码库对外展示时使用的项目名称。这个仓库的目标不是复刻私有自动驾驶系统，而是把“规划核心”整理成一个适合公开展示、GitHub 浏览和简历描述的完整项目：场景表示、条件轨迹生成、评估体系和可视化产物。

![RouteDiffuser demo plot](outputs/portfolio_demo/prediction_plot.png)
![RouteDiffuser scenario gallery](outputs/portfolio_demo/scenario_gallery.png)

## 这个仓库解决什么问题

大多数自动驾驶项目都不能公开真实数据、仿真平台和内部部署链路，所以公开仓库往往只能停留在“模型片段”。这个仓库聚焦那些真正可以公开、也真正能体现工程能力的部分：

- 统一的规划场景 schema：ego、邻车、车道线、route polyline 和 mask
- 基于 route prior 的残差扩散规划策略
- 受更大规划项目启发的多分辨率扩散噪声
- 候选轨迹评分与选择，而不是默认拿第一个 sample
- 可复现的 open-loop 评估、候选轨迹图、多场景画廊和报告产物

## 当前能力

- 已实现条件扩散规划模型，围绕路线先验做未来轨迹去噪。
- 已实现 DDPM 风格训练、迭代采样和首帧锚定。
- 已实现多候选轨迹打分，综合 route、clearance 和 comfort proxy 做选择。
- 已实现结构化 synthetic 场景，覆盖直行、左变道、右变道和缓弯。
- 已补齐公开 CLI、数据 manifest、正式评估 report schema 和作品集 demo 脚本。

## 快速开始

```bash
pip install -e .[dev]
python scripts/prepare_dataset.py --output outputs/manifests/route_diffuser_synthetic_train.json
python scripts/demo_planner.py
```

安装后也可以直接使用命令别名：

- `route-diffuser-prepare`
- `route-diffuser-stats`
- `route-diffuser-export-npz`
- `route-diffuser-export-onnx`
- `route-diffuser-check-onnx`
- `route-diffuser-benchmark`
- `route-diffuser-rollout`
- `route-diffuser-compare-scorer`
- `route-diffuser-ablations`
- `route-diffuser-failures`
- `route-diffuser-train`
- `route-diffuser-infer`
- `route-diffuser-eval`
- `route-diffuser-demo`

命令手册见：

- [docs/commands.md](docs/commands.md)
- [docs/artifacts.md](docs/artifacts.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/release_checklist.md](docs/release_checklist.md)
- [docs/ablations.md](docs/ablations.md)

上面的 demo 命令会在 `outputs/portfolio_demo/` 下生成一套完整展示产物：

- `prediction_plot.png`
- `candidate_trajectories.png`
- `scenario_gallery.png`
- `evaluation_report.json`
- `evaluation_report.md`
- `portfolio_summary.json`
- `portfolio_summary.md`
- `demo_checkpoint.pt`
- `predictions.pt`

## 命令入口

公开入口：

- `python scripts/prepare_dataset.py`
- `python scripts/compute_dataset_stats.py`
- `python scripts/compare_scorer.py`
- `python scripts/run_ablation_matrix.py`
- `python scripts/analyze_failures.py`
- `python scripts/train_planner.py`
- `python scripts/infer_planner.py`
- `python scripts/eval_planner.py`
- `python scripts/demo_planner.py`

兼容旧入口：

- `python scripts/train_diffusion.py`
- `python scripts/infer_scene.py`
- `python scripts/eval_diffusion.py`
- `python scripts/demo_portfolio.py`

## Demo 产物

作品集 demo 会训练一个小模型、采样未来轨迹、输出 open-loop 指标，并生成适合放在 GitHub 首页或简历项目页里的总结材料。

- 项目路线图：[ROADMAP.md](ROADMAP.md)
- 数据准备脚本：[scripts/prepare_dataset.py](scripts/prepare_dataset.py)
- 主 demo 脚本：[scripts/demo_planner.py](scripts/demo_planner.py)
- 作品集总结：[outputs/portfolio_demo/portfolio_summary.md](outputs/portfolio_demo/portfolio_summary.md)
- 结构化评估报告：[outputs/portfolio_demo/evaluation_report.md](outputs/portfolio_demo/evaluation_report.md)
- 候选轨迹图：[outputs/portfolio_demo/candidate_trajectories.png](outputs/portfolio_demo/candidate_trajectories.png)
- 多场景画廊：[outputs/portfolio_demo/scenario_gallery.png](outputs/portfolio_demo/scenario_gallery.png)

## 架构概览

```text
结构化场景生成器 / 后续真实数据适配器
  -> 统一场景张量
  -> scene encoder
  -> route prior 构建
  -> 条件扩散解码器（1D U-Net）
  -> 迭代轨迹去噪
  -> 候选轨迹评分与选择
  -> open-loop 指标与调试可视化
```

## 仓库结构

- `planner/datasets/`：场景 schema、synthetic 数据和数据工厂
- `planner/datasets/adapters/`：可插拔数据适配层，当前包含 synthetic 和公开 NPZ 格式
- `planner/models/`：scene encoder 和 diffusion decoder
- `planner/diffusion/`：噪声调度与反向扩散工具
- `planner/inference/`：锚定和候选打分逻辑
- `planner/trainers/`：训练与评估入口
- `planner/reports/`：结构化评估报告 schema
- `planner/visualization/`：轨迹绘图工具
- `scripts/`：数据准备、训练、推理、评估和 demo 入口
- `configs/model/`：基础模型配置，以及 scorer / encoder 对比配置
- `docs/commands.md`：命令手册
- `docs/artifacts.md`：产物说明文档
- `docs/architecture.md`：模块架构说明
- `docs/release_checklist.md`：发布检查清单
- `docs/ablations.md`：实验矩阵和 scorer 对比说明
- `ROADMAP.md`：项目完善路线图

## 适合怎么描述到简历里

- 自动驾驶规划项目，重点展示 route-conditioned future trajectory generation。
- 使用 PyTorch 实现条件扩散模型和 1D U-Net 轨迹解码器。
- 包含统一场景接口、synthetic 场景生成、候选轨迹排序、分场景评估和可视化产物。

## 当前边界

这个仓库有意保持 `planner-first`，而不是做成一个私有自动驾驶系统的空壳复刻。

已经包含：

- 场景建模
- 扩散规划
- synthetic 数据
- manifest / adapter 骨架
- 候选评分
- 评估报告
- 可视化和作品集产物

暂不包含：

- 私有数据适配器
- 私有仿真服务
- TensorRT/内部部署插件
- RL fine-tuning 链路

这种边界是有意设计的。它能让项目在公开环境里保持完整、可运行、可解释，而不是依赖无法共享的基础设施。
