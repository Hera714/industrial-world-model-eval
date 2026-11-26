# Week 8 Agent 协议（初版）

## 职责与数据流

用户提出自然语言任务；LLM 产生结构化意图和工具调用建议。受控代码以固定模板生成候选配置，先执行 Schema、白名单、资源约束及 Week 7 完整预检，再启动 Runner。环境推进、预测、指标计算与结果审核均由确定性代码完成；LLM 可解释审核结果但不能修改数值或将失败标记成成功。

## 输入和输出

输入包括任务文本、明确的模型和执行配置；输出包括生成配置、配置摘要、工具调用和结果、原始 run 路径、审核结果、报告及失败上下文。记录使用的 LLM 模型名、版本、解码参数和提示词版本。记录不得包含凭据。

## 白名单和上限

`configs/agent/policy.yaml` 冻结 BOPTEST 案例、模型类型、工具和资源上限；模型只能选择已有检查点，不能提交任意路径、URL、shell 命令或修改白名单。动作范围、单位、数据分块和模型输入信息集沿用 Week 7 配置契约。用户要求超出支持范围时返回明确错误，不能悄悄改写意图。

## 工具与失败

四个工具分别列出环境、验证配置、运行 benchmark、审核并汇总结果。工具需提供结构化参数、结果及错误码；区分配置缺失、非法环境、数值无效、超时和服务异常。执行前检查，执行后保存状态、轨迹与失败信息；advance 失败后不自动重试。LLM 使用的文本和外部结果均不是新的执行权限。

所有工具返回统一对象 `{"ok": bool, "result": object 或 null, "error": object 或 null}`；失败时 `error` 包含 `code`、异常 `type`、原始 `message` 和 `suggestion`。参数的 JSON Schema 在 `agents/schemas.py` 中，拒绝额外字段；更细的冻结配置与资源限制由 `validate_candidate` 检查。

| 工具 | 参数 | 成功结果 |
| --- | --- | --- |
| `list_envs` | 空对象 | `environments` 列表 |
| `validate_config` | `config` 对象 | `valid` 和已验证的 `models` |
| `run_benchmark` | `config` 对象 | `run_directory` |
| `summarize_results` | `run_directory` 字符串 | `passed` 与逐项 `checks` |

统一错误代码包括 `TIMEOUT`、`INVALID_ENVIRONMENT`、`MISSING_FIELD`、`INVALID_PREDICTION` 和 `EXECUTION_ERROR`；后者覆盖未单独分类的错误，原始消息保留以便定位。ReAct 还在轨迹中记录跳过校验、重复执行和审核非本会话 run 等门禁错误。

## 复现与审批

沿用固定检查点、数据、种子与哈希；每次运行保留配置快照、版本及原始产物。高预算或协议规定需人工批准的任务，由状态机记录具体配置和批准结果。批准只针对该次配置，不代替校验。规则流程可作为离线测试，不宣称为真实 LLM 演示。

## LangGraph 人工审批与持久化

当调用方将 `require_approval` 设为 true，或模型数量超过 `configs/agent/policy.yaml` 中的 2 个阈值时，图会在执行前使用 LangGraph interrupt 返回候选配置和 `config_sha256`；这一阈值是项目的预算规则，并非通用风险标准。普通流水线和 ReAct 循环遇到超过阈值的配置会拒绝执行并要求使用审批图。审批方应查看具体配置，再以 `Command(resume={"approved": True, "config_sha256": 摘要})` 恢复；拒绝或摘要不一致不会执行。执行节点再次检查摘要，运行工具也要求绑定摘要并重新验证白名单和输入哈希；摘要作为内部函数参数传递，不在模型可调用的工具 Schema 中。

`build_graph` 接受外部 checkpointer。使用 SQLite 的 `SqliteSaver` 可在进程重启后恢复；同一线程使用相同 `thread_id`，并保持数据库文件。默认内存 checkpointer 仅适用于进程内测试。SQLite 跨连接恢复、错误摘要拒绝已有测试；真实 LLM 会话经用户批准后已完成 BOPTEST 和审核，独立复核见 `reports/week8/graph_approved_verification.json`。
