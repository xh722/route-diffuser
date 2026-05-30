# RouteDiffuser 发布检查清单

每次发布代码、刷新 demo、更新模型 checkpoint 或整理简历材料前，都按这份清单检查一次。

## 发布范围

先确认这次发布属于哪一类：

- [ ] 仅文档更新
- [ ] 代码 + 文档更新
- [ ] demo / portfolio artifact 刷新
- [ ] 模型 checkpoint 刷新
- [ ] 里程碑版本更新

## 工作区

- [ ] `git status` 中的所有文件都已确认。
- [ ] 没有误提交临时输出、缓存文件或草稿文件。
- [ ] 大文件只在确实需要作为展示产物时才提交。
- [ ] 新增配置、文档、测试和代码保持一致。

## 命令入口

- [ ] `docs/commands.md` 与 `scripts/` 下的公开命令一致。
- [ ] README 中列出的脚本仍然存在。
- [ ] `pyproject.toml` 中新增的 CLI alias 已写入文档。
- [ ] 兼容旧入口的 wrapper 仍然指向正确模块。

## 数据层

- [ ] manifest 生成命令仍可运行。
- [ ] synthetic 数据配置仍是默认可运行路径。
- [ ] NPZ bridge 的字段和 shape 文档仍然准确。
- [ ] 新增数据适配器时同步补充配置示例和测试。

## 模型与训练

- [ ] 模型配置字段与 `DiffusionPlannerConfig` 一致。
- [ ] 新增 scorer / encoder / candidate strategy 已写入 `docs/ablations.md`。
- [ ] checkpoint 命名清晰。
- [ ] 训练日志格式没有无意变更。

## 评估层

- [ ] 评估报告 schema 的变更是有意的。
- [ ] 新增指标已写入文档和 portfolio summary。
- [ ] metric 命名在 report、registry、ablation 和 demo 中保持一致。
- [ ] safety 指标明确区分 point collision 和 box collision。

## 部署层

- [ ] ONNX export 命令仍在文档中。
- [ ] parity check 命令仍在文档中。
- [ ] benchmark 命令仍在文档中。
- [ ] 导出产物文件名稳定。

## 闭环 Rollout

- [ ] rollout 命令仍在文档中。
- [ ] rollout trace、summary、plot 产物说明仍准确。
- [ ] `collision_flags` 的含义没有静默变化。
- [ ] box collision 和 point collision 指标都能被解释。

## Demo 与简历材料

- [ ] `outputs/portfolio_demo/` 反映当前项目能力。
- [ ] `portfolio_summary.json` 和 `.md` 与当前代码指标一致。
- [ ] README 首页图片存在并可渲染。
- [ ] scenario gallery 和 candidate trajectory 图与当前 pipeline 对应。
- [ ] `docs/portfolio_case_study.md` 与当前代码能力一致。

## 文档

- [ ] `README.md` 是中文主文档。
- [ ] `README.zh-CN.md` 与主 README 保持一致或明确说明差异。
- [ ] `docs/artifacts.md` 覆盖当前输出目录和文件名。
- [ ] `docs/architecture.md` 与代码结构一致。
- [ ] `docs/ablations.md` 覆盖当前实验配置。
- [ ] `ROADMAP.md` 如实反映当前进展。

## 验证

根据发布范围选择合适的验证：

- [ ] `python -m compileall planner scripts tests`
- [ ] targeted unit tests
- [ ] targeted CLI smoke commands
- [ ] demo artifact refresh commands
- [ ] ONNX export / parity smoke

在 commit message、PR 描述或 release note 中写明实际运行过什么。

## 最终检查

- [ ] commit message 能概括改动。
- [ ] 文档入口清楚。
- [ ] 新读者打开仓库后能理解项目是什么、怎么跑、怎么评价。
- [ ] 简历中引用这个项目时，有明确的技术亮点和可展示产物。
