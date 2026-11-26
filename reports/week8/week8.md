# Week 8 当前进度

## 已完成的基础组件

- Agent 边界、白名单、工具参数契约和冻结 smoke 配置映射。
- 规则审核：结果文件、哈希、运行状态、配置单位、矩阵及有效预测计数。
- 错误分类、Agent 编排、LangGraph 节点与 SQLite 跨连接恢复测试。
- 20 条测试用例与规则基线离线记录；自动报告和面试材料。
- Agent 相关测试共 53 项通过（其中包含模拟工具与模型的测试）。

## 验收状态与边界

真实 LangGraph 会话已由用户批准后恢复，并完成 BOPTEST 和规则审核；不能把模拟响应或离线规则结果当作真实模型成绩。

Qwen 本地模型权重已下载并通过 SHA256 验证，真实端到端演示已完成一次。已有结果和资料路径详见 `docs/architecture.md`、`reports/week8/automatic_analysis.md` 与 `docs/interview/`。

ReAct、PaperBench 与 AgentSuite 的方法迁移和复现边界见 `docs/agent_methods.md`。

真实人工审批会话的节点顺序、配置摘要和 run 已独立复核，记录见 `reports/week8/graph_approved_verification.json`；运行日志见 `reports/week8/graph_approved_execution.log`。

## 主要实验结果（不同环境不可直接排名）

| 阶段 | 实验和条件 | 代表性结果 | 可追溯来源 |
| --- | --- | --- | --- |
| Week 5 | 自建二阶热工环境，18 个测试 episode，真实起点后自由递推；记录的未来天气作为离线条件 | 室温 20 步末端 RMSE：Persistence 0.863 °C、Linear 0.0345 °C、直接 MLP 0.0531 ± 0.0074 °C、残差 MLP 0.0421 ± 0.0012 °C；MLP 的 ± 为 3 个训练种子样本标准差 | [Week 5 定义与完整表](../week5/week5.md)、[逐步结果](../week5/results/rollout/metrics_key_horizons.csv) |
| Week 6 | BOPTEST，三个主要参考日期；20 步是 5 小时末端预测，控制成本为另一个闭环实验 | ARX：温度 RMSE 0.716 °C、成本 61.00；Persistence：1.014 °C、成本 171.74。数字是日期均值，不能据此作单因素因果结论 | [Week 6 协议与表](../week6/week6.md)、[逐日期汇总](../week6/results/summary/prediction_vs_control_reference.csv) |
| Week 6 案例 | 同一 `id_test_day340` 的配对观察 | Persistence 预测 RMSE 0.667 °C 低于 `mlp_seed43` 的 0.753 °C，但闭环成本 382.97 高于 124.44；两种任务的信息集和动作不同 | [案例与原始指标](../week6/results/summary/prediction_control_counterexamples.csv)、[分析及轨迹](../week6/results/consistency/cases.md) |
| Week 7 | 真实 BOPTEST 短评测及批量恢复检查 | 短评测评分 8 步；批量验证 4 个任务、实际峰值并发 2，恢复检查通过。这是工程验收，不是模型精度比较 | [Week 7 报告](../week7/week7.md)、[批量验证记录](../week7/batch_verification.json) |

Week 5 的加热功率控制、自建环境和 Week 6 的 BOPTEST 温度设定值控制不同，数据和代价也不同。上表数字只在各自协议内解释；闭环成本是该实验的加权评分，不能直接解释为电费。

## 失败案例与项目局限

- 真实 BOPTEST 曾在选择案例时超时；[失败分析](automatic_failure_example.md)保留原始错误和阶段。状态推进失败不能盲目重试，以免重复动作。
- 自主工具联调前两次未正常结束，其中一次重复运行；后续加入执行门禁并保留[原始轨迹](react_real_execution_second_attempt.json)。当前 20 条开发用例上的严格首步选择为 7/20，安全检查不可省略。
- BOPTEST 是单案例仿真，日期数量有限；公开观测不等于完整建筑状态。模型可能产生无效能耗或触发 fallback，当前机制不是现场安全认证。
- 20 条 Agent 用例参与设计，人工语义复核尚未完成；不宜把结果当作独立测试集性能或真实建筑部署证明。

真实本地 Qwen 与 BOPTEST 端到端成功会话：`reports/week8/sessions/20261125T161119997809Z_c54402617b/session.json`。20 条真实模型离线测试和规则对照见 `agent_eval.md`；对照集参与过规则设计，不能视为独立泛化评估。

## 自主工具选择的额外检查

在相同的 20 条开发用例上，实际调用本地 Qwen，让它在没有历史观察时提出第一个工具。严格规则要求可执行请求先选择 `validate_config`，其他请求选择 `finish`（拒绝或澄清）；调整提示后结果为 **7/20**，调整前为 **5/20**。逐条响应及模型哈希分别见 `reports/week8/tool_choice_first_action.json` 和 `reports/week8/tool_choice_first_action_before_prompt_change.json`。例如模型有时验证不应执行的请求。此数字只衡量严格的首步选择，不衡量完整工具序列；把 `list_envs` 用于澄清也未被该严格评分计为正确。它与固定程序编排下的配置生成成绩不同，也不能当作独立测试集泛化结果。

针对该风险，ReAct 执行层现在要求先成功校验完全相同的配置，且用户原文明确包含 BOPTEST 与模型名称，才能调用运行工具；运行工具自身仍独立执行白名单校验。测试涵盖跳过校验、校验后换配置及错误环境。这个保守门禁会拒绝某些名称不明确的自然语言表达，不代表小模型能够可靠自主执行。超过 2 个模型的配置在 LangGraph 自动进入人工审批，普通流水线和 ReAct 会提示转用审批图。

真实自主行动—观察联调中，前两次执行和审核成功后模型未结束，第二次还重复执行了同一配置。现已加入防重复执行和本次 run 来源校验，缩短模型接收的审核观察，同时保留完整审核记录。改进后多次完成 `validate_config → run_benchmark → summarize_results → finish`；真实结果、决策提示及前两次失败轨迹见 `reports/week8/react_real_execution*.json`。这些成功不代表其他请求可靠。

## 自动分析报告复现

`agents/report_generator.py` 从 Week 6 已有产物生成图表、成本示例、预测—控制反例和指定 run 的失败记录链接；缺失数据标为缺失，文件附带 SHA256。以下命令使用已完成的真实示例：

```bash
python -m agents.report_generator --run-directory reports/runs/20261125T161124183844Z_boptest_smoke_d1f6120092c3
```

输出为 `reports/week8/automatic_generated.md`。真实超时案例另见 `reports/week8/automatic_failure_example.md`；报告提取错误类型、原始消息和执行阶段，不声称自动确定失败的因果原因。

规则基线已提供 `scripts/evaluate_agent_rules.py --execute` 真实执行选项；它仅对冻结用例中明确允许且模型匹配的任务运行，并在服务错误时保存部分结果后停止。相同预算下的真实 BOPTEST 对照已完成：两组各 6/6 合法任务运行并通过审核。口径、耗时及逐条记录见 [Agent 对照](agent_eval.md)。

## 容器复现

基于 Week 7 评测镜像和 `deployment/agent/requirements.lock` 新建 CPU 容器，挂载冻结模型与数据并连接已启动的 BOPTEST。计时真实会话 `reports/week8/sessions/20261126T032650861248Z_84ffb07296/` 完成，宿主机通过项目相对路径重新核验，结果见 `reports/week8/container_timed_verification.json`。已准备资产后，从容器启动到审核完成的完整运行用时 92.96 秒，记录见 `reports/week8/container_elapsed_seconds.txt`；不含镜像构建与资产下载，其他环境耗时可能不同。运行命令和前提见 README。
