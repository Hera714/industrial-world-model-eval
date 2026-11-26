# Week 8 方法参考与迁移

## ReAct：观察后再决定

[ReAct 原论文](https://arxiv.org/abs/2210.03629)研究交错生成推理与行动，利用外部环境的观察修正后续步骤；原文在问答、事实核验和交互环境中实验。本项目借鉴行动—观察循环：本地模型提出一个白名单工具，程序执行后将结果提供给模型，再由模型选择下一步。完整动作和原始观察保存在 `reports/week8/react_real_execution.json`；模型只接收审核结果的决策摘要，完整审计详情仍在日志中。

`agents/react.py` 限制轮数，执行前要求同配置校验、请求中的平台与模型匹配，并阻止重复运行和引用本会话以外的 run。`scripts/test_react_correction.py` 用明确标记的注入故障检查模型能否重新校验；该实验不运行真实仿真。`scripts/run_react_local.py` 则实际运行 BOPTEST。前两次真实联调未能正常结束，改进后两次完成。我们没有复现原论文的任务、提示设计或评分，也不记录或声称获取模型私有推理过程。

## PaperBench：把任务拆成可核验项

[PaperBench](https://proceedings.mlr.press/v267/starace25a.html)面向论文复现，将复杂任务分解成可评分的层级细则。本项目借鉴这种分解思想，但评测的是楼宇 benchmark Agent 而非论文复现：`data/agent_eval/week8_cases.json` 为每条请求保存预期动作、模型和配置、工具、结果 rubric；`evaluators/agent.py` 分别计算配置正确、工具选择、错误识别和真实运行审核通过，并明确分母。原始事件和分数保存在 `reports/week8/`。

示例：`agent_01` 要求 persistence 和 arx，必须先校验、再在 BOPTEST 中执行、最后审核；真实对照记录见 `rule_real_evaluation.json` 与 `local_llm_real_evaluation.json`。指标程序而非 LLM 决定分数。我们没有复现 PaperBench 的论文集、层级评分规模或其自动裁判，不能将本项目结果与 PaperBench 分数比较。

## AgentSuite / COBA：检查评测本身

[AgentSuite](https://proceedings.mlr.press/v306/suh26a.html)提出的 COBA 从用户、环境、标准答案、评测四部分识别 benchmark 问题。本项目仅迁移这四个检查维度，不声称复现其完整分类体系或混合检测器：

| 检查部分 | 本项目对应证据 | 尚存局限 |
| --- | --- | --- |
| 用户指令 | `week8_cases.json` 保存 20 条原文与预期 | 歧义和答案合理性仍需独立人工复核 |
| 环境 | 冻结 BOPTEST 案例、模型和执行预算，记录真实 run | 模拟器状态与服务负载仍会变化 |
| 标准答案 | 每条请求记录预期动作、模型及 rubric | 开发集参与规则与提示设计，不是独立测试集 |
| 评测程序 | `agent_dataset_audit.json` 检查结构，逐条轨迹可重新评分 | 结构检查不能证明语义正确；严格首步评分可能把合理澄清判错 |

`reports/week8/dataset_human_review.md` 是待人工复核清单，不代表独立专家已经审核。本项目没有对原始 AgentSuite 基准执行 COBA，也没有复现论文的专家一致性结果。

## 结果应怎样使用

同一开发集上，固定规则编排与本地模型配合白名单的真实 BOPTEST 运行均为 6/6；它们的 14 条拒绝/澄清请求不执行仿真。独立测量的自主工具**首步**严格匹配率为 7/20，不能把固定编排的成功率当成自主工具能力。具体结果、耗时口径和限制见 `reports/week8/agent_eval.md`。
