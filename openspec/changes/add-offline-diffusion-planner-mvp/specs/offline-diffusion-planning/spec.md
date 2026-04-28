## ADDED Requirements

### Requirement: Canonical scene input contract
The system SHALL define a canonical scene input contract for offline planning that includes ego state, neighbor history, lane context, route context, and validity masks.

#### Scenario: Normalize dataset-specific samples into one planner schema
- **假设** 输入样本来自某个具体数据源，字段命名和布局与核心模型无关
- **当** 数据适配层将样本送入规划器
- **则** 适配层 SHALL 输出统一的场景结构
- **且** 该结构至少包含 `ego`、`neighbors`、`lanes`、`route_lanes` 和 mask 信息

### Requirement: Multi-modal scene encoding
The system SHALL encode dynamic agents, lane geometry, and route context before diffusion decoding.

#### Scenario: Encode scene context for one planning batch
- **假设** 一个批次内包含 ego、邻居历史、车道和路线信息
- **当** 规划器执行编码阶段
- **则** 系统 SHALL 为扩散解码器输出融合后的上下文特征
- **且** 无效 token 不应对有效特征产生未受控影响

### Requirement: Conditional future trajectory generation
The system SHALL generate a future ego trajectory conditioned on scene context and diffusion timestep.

#### Scenario: Predict trajectory noise during training
- **假设** 一条真实未来轨迹被加噪到某个扩散时间步
- **当** 模型接收带噪轨迹、扩散时间步和场景条件
- **则** 系统 SHALL 输出与轨迹形状一致的噪声预测张量

#### Scenario: Generate one future sample during inference
- **假设** 一个场景样本和一个初始随机噪声
- **当** 推理流程执行迭代去噪
- **则** 系统 SHALL 输出一条未来 ego 轨迹
- **且** 输出维度必须与配置的未来时域长度一致

### Requirement: First-state anchoring
The system SHALL preserve the current ego state at the first predicted timestep during inference.

#### Scenario: Keep the predicted trajectory attached to the current state
- **假设** 推理输入中包含当前 ego 状态
- **当** 系统生成未来轨迹
- **则** 输出轨迹的第一个状态 SHALL 与当前 ego 状态对齐
- **且** 后续去噪步骤不应破坏该约束

### Requirement: Multi-sample planning inference
The system SHALL support generating multiple trajectory samples from the same scene by changing the inference noise seed or sample index.

#### Scenario: Produce multiple candidate futures for one scene
- **假设** 同一个场景被请求生成多个候选轨迹
- **当** 推理脚本使用不同的噪声初始化或采样编号
- **则** 系统 SHALL 返回多条候选未来轨迹
- **且** 候选结果应保留与场景样本的对应关系
