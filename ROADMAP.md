# RouteDiffuser 路线图

这份路线图用于把 `RouteDiffuser` 打磨成一个公开、可复现、适合社招简历展示的自动驾驶规划项目。目标不是复刻公司内部系统，而是把规划核心能力整理成外部读者能理解、能运行、能评价的工程作品。

## 项目目标

`RouteDiffuser` 的目标是提供一个完整的 planner-core 项目：

- 可复现的训练、推理、评估、导出和 demo 命令。
- 统一的场景 schema 和数据适配边界。
- 至少一条公开格式数据路径。
- 结构化评估报告和可视化产物。
- ONNX 导出、parity 检查和 benchmark。
- 轻量 closed-loop rollout。
- 面向简历和 GitHub 展示的案例文档。

## V1 完成标准

V1 版本完成时，应满足：

1. 新机器 clone 仓库后，可以按 README 安装依赖并运行 train/eval/infer/export/demo。
2. 支持 synthetic 数据和至少一种公开格式数据适配路径。
3. evaluation 输出 JSON 和 Markdown 报告，包含 overall、scenario、candidate-set、安全和舒适性指标。
4. denoiser core 可以导出 ONNX，并通过 PyTorch parity check。
5. 具备轻量 closed-loop rollout，输出 trace、summary 和 plot。
6. 主要 pipeline 有测试和 CI smoke check。
7. 文档能让面试官快速理解项目价值、架构和边界。

## V1 非目标

以下内容不纳入公开 V1：

- 私有 protobuf / service stack。
- 公司内部日志格式和私有数据适配器。
- TensorRT 或内部部署插件。
- 多机分布式训练。
- 生产级 RL serving。
- 等价于公司内部仿真的完整 closed-loop 平台。

## 当前能力地图

| 能力 | 当前状态 | 说明 |
| --- | --- | --- |
| 数据 schema | 已完成 | canonical scene tensors 覆盖 ego、neighbor、lane、route 和 mask |
| 数据适配 | 已完成 | synthetic + NPZ bridge + manifest |
| 扩散模型 | 已完成 | route-prior residual diffusion + conditional 1D U-Net |
| 候选选择 | 已增强 | heuristic / learned hybrid scoring + scorer diagnostics |
| 结构化候选 | 已增强 | noise、drift、mixed、route-anchor candidate strategy |
| 安全指标 | 已增强 | point collision + oriented-box collision |
| 评估报告 | 已完成 | JSON / Markdown report + scenario breakdown |
| 部署导出 | 已完成 | ONNX export、parity、benchmark |
| 闭环 rollout | 已完成 | lightweight receding-horizon rollout |
| 实验管理 | 已完成 | scorer/encoder ablation + registry |
| 简历材料 | 已完成 | portfolio case study + demo artifacts |

## 阶段规划

### P0：公开项目骨架

目标：让项目从 demo 脚本变成可运行的公开仓库。

已完成：

- [x] 增加 `planner/datasets/adapters/`，定义稳定数据适配边界。
- [x] 增加 `scripts/prepare_dataset.py`，支持 manifest 生成。
- [x] 使用 JSON manifest 管理数据切片。
- [x] 把 synthetic 数据配置从模型配置中拆开。
- [x] 评估输出结构化 report contract。
- [x] 增加 train/infer/eval/demo smoke 脚本。
- [x] 增加 CI 路径。
- [x] 增加命令、产物和架构文档。

### P1：公开 V1 完整能力

目标：具备数据、评估、部署、闭环和文档闭环。

已完成：

- [x] 实现公开 NPZ bridge format。
- [x] 增加数据统计缓存。
- [x] 统一 train/eval/infer CLI。
- [x] 增加 ONNX export。
- [x] 增加 ONNX parity check。
- [x] 增加 inference benchmark。
- [x] 增加 lightweight closed-loop rollout。
- [x] 增加 rollout metrics 和 trace plot。
- [x] 扩展 README、docs 和 release checklist。

### P2：模型和研究能力升级

目标：让项目不只是工程包装，还具备可研究的实验轴。

已完成：

- [x] 增加 learned trajectory scorer head。
- [x] 增加 reward-aware scorer target mode。
- [x] 增加 scorer candidate strategy：noise、drift、mixed、route-anchor。
- [x] 增加 attention-based scene fusion ablation。
- [x] 增加 scorer diagnostics：heuristic regret、learned preference regret、agreement rate。
- [x] 增加 oriented-box collision，并接入 open-loop 和 rollout。
- [x] 增加 scorer/encoder ablation matrix。
- [x] 增加 failure analysis 和 registry。

后续可继续做：

- [ ] 在公开真实数据上验证 NPZ bridge 流程。
- [ ] 增加 scorer calibration report。
- [ ] 增加 failure-case gallery。
- [ ] 增加 model card / data card。
- [ ] 进一步升级 scene encoder 为 ego-query cross attention。
- [ ] 在 rollout 稳定后考虑 reward modeling 或 offline RL fine-tuning。

## 推荐后续优先级

1. **真实公开数据示例**：用 NPZ bridge 准备一个可公开的小样例，让项目说服力更强。
2. **Scorer calibration report**：比较 heuristic best、learned best、selected candidate 和 oracle rank。
3. **Failure gallery**：为 collision、route deviation、comfort violation 自动出图。
4. **Model card**：说明训练数据、指标、适用范围和限制。
5. **CI smoke command**：在 CI 中加入一个超小规模 CLI smoke，进一步证明可运行性。

## 主要风险

- 数据范围膨胀：过早支持太多格式会拖垮维护成本。
- 评估指标频繁改名：会影响报告、registry 和文档稳定性。
- 仿真过度设计：公开项目应该保持轻量 rollout，而不是复刻私有服务。
- 模型复杂度先行：在数据和评估不稳时堆模型结构，收益不高。

## 简历定位

这个项目适合被描述为：

- 自动驾驶轨迹规划项目，重点是 route-conditioned future trajectory generation。
- 使用 PyTorch 实现条件扩散模型、1D U-Net 去噪器和多候选轨迹选择。
- 包含统一场景接口、数据适配、评估报告、碰撞安全指标、消融实验、ONNX 导出和闭环 rollout。
