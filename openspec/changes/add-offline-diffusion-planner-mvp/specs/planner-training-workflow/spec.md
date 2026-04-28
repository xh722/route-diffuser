## ADDED Requirements

### Requirement: Normalization and angle continuity handling
The training workflow SHALL normalize planner inputs and targets into a stable model space and represent heading continuously.

#### Scenario: Convert raw heading to continuous features
- **假设** 输入或标签中包含角度形式的航向信息
- **当** 预处理流程为模型准备训练张量
- **则** 系统 SHALL 将航向转换为连续表示，例如 `cos/sin`
- **且** 仅在可视化或最终输出边界再转回角度

#### Scenario: Map scene values into normalized ranges
- **假设** 原始输入来自物理量空间，例如位置、速度或加速度
- **当** 训练流程准备模型输入
- **则** 系统 SHALL 应用可复现的归一化逻辑
- **且** 推理输出必须支持反归一化回物理空间

### Requirement: Diffusion noise-prediction objective
The training workflow SHALL optimize the planner as a conditional diffusion model using noisy future trajectories as supervision.

#### Scenario: Train on randomly sampled diffusion timesteps
- **假设** 一个批次包含真实未来轨迹标签
- **当** 训练循环执行一次前向传播
- **则** 系统 SHALL 随机采样扩散时间步
- **且** 对真实未来轨迹注入对应噪声
- **且** 以真实噪声和预测噪声之间的损失作为优化目标

### Requirement: Reproducible experiment execution
The training workflow SHALL support reproducible runs through explicit configuration, checkpointing, and seed control.

#### Scenario: Resume or compare experiments consistently
- **假设** 用户使用同一配置和随机种子重复运行训练
- **当** 训练脚本执行
- **则** 系统 SHALL 记录配置、种子和检查点元数据
- **且** 支持从已保存检查点恢复训练

### Requirement: Open-loop evaluation artifacts
The project SHALL provide open-loop evaluation outputs that are sufficient for debugging planner quality before closed-loop integration.

#### Scenario: Evaluate predicted futures against ground truth
- **假设** 已训练模型和一批带标签的场景样本
- **当** 评估脚本运行
- **则** 系统 SHALL 计算轨迹误差类指标
- **且** 输出可供人工检查的可视化或结构化结果

#### Scenario: Detect route or feasibility regressions during evaluation
- **假设** 预测轨迹虽然数值误差不高，但存在明显偏离路线或起点漂移的问题
- **当** 评估脚本分析预测结果
- **则** 系统 SHALL 提供额外的规则检查或调试产物
- **以便** 用户识别仅靠平均误差无法发现的问题
