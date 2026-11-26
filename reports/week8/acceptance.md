# Week 8 验收清单

本清单依据当前文件、测试及真实运行记录整理。状态“待核验”不计作已完成；模型和数据需先按原计划准备好。

| 任务 | 当前证据 | 状态与限制 |
| --- | --- | --- |
| P0-1 职责、白名单和预算 | `docs/agent_protocol.md`、`configs/agent/policy.yaml` | 已实现；指标由确定性代码计算 |
| P0-2 工具契约 | `agents/schemas.py`、`agents/tools.py`、`tests/test_agent_schemas.py` | 已实现四个工具和参数白名单 |
| P0-3 自然语言配置 | `agents/local_llm.py`、`agents/config_generator.py`、`configs/agent/local_model_manifest.json` | 本地 Qwen 真实生成并校验；模型不可任意修改配置 |
| P0-4 规则审核 | `agents/result_auditor.py`、`tests/test_agent_auditor.py`、`tests/test_agent_kpi_audit.py` | 检查完成状态、哈希、单位、矩阵、预测有效性和部分跨文件一致性 |
| P0-5 错误处理 | `agents/errors.py`、`tests/test_agent_errors.py` | 缺字段、非法环境、非有限预测、超时均有分类和建议 |
| P0-6 真实端到端 | `reports/week8/sessions/20261126T032650861248Z_84ffb07296/`、`reports/week8/container_timed_verification.json` | 真实 LLM、BOPTEST 和确定性审核通过；会话保留输入、配置与 run 路径 |
| P0-7 文档与演示 | `README.md`、`docs/architecture.md`、`reports/week8/week8.md` | CPU 容器已准备资产后计时 92.96 秒；首次安装和下载不计入 |
| P0-8 面试材料 | `docs/interview/five_minutes.md`、`ten_minutes.md`、`questions.md` | 已提供两份讲稿与十问；结果及局限与现有证据对齐 |
| P1-9 LangGraph | `agents/graph.py`、`agents/state.py`、`tests/test_agent_graph.py`、`tests/test_agent_graph_sqlite.py`、`reports/week8/graph_approved_verification.json` | 中断、摘要绑定、拒绝及持久化恢复已测试；用户批准后真实会话完成且独立审核通过 |
| P1-10 评测用例 | `data/agent_eval/week8_cases.json`、`reports/week8/agent_dataset_audit.json` | 20 条含 rubric；结构检查通过，语义尚待独立人工复核，开发集参与过设计 |
| P1-11 Agent 指标 | `evaluators/agent.py`、`reports/week8/agent_eval.md` | 真实配置/拒绝/运行分母与结果已记录；自主工具只单独衡量首步 |
| P1-12 自动分析报告 | `agents/report_generator.py`、`reports/week8/automatic_generated.md`、`automatic_failure_example.md` | 图和表有来源与哈希；缺失数据明示，错误原因取自日志而不臆测因果 |
| P2-13 ReAct | `agents/react.py`、`reports/week8/react_real_execution.json`、`react_validation_correction.json` | 校验故障注入实验和真实工具调用已记录；不声称复现原论文基准 |
| P2-14 参考方法迁移 | `docs/agent_methods.md`、`reports/week8/dataset_human_review.md` | 保存 rubric 和组件检查；不声称复现 PaperBench 或 AgentSuite |
| P2-15 规则与 LLM 对照 | `reports/week8/agent_eval.md`、`rule_real_evaluation.json`、`local_llm_real_evaluation.json`、`comparison_reaudit.json` | 相同用例与仿真预算，真实有效请求各 6/6；12 个 run 已独立复核，提示调用精确总数未可靠记录 |

最近一次 Agent 相关单元测试：53 项通过（包括模拟和离线测试），静态检查通过。结果见终端执行记录；真实运行证据应以各会话和 run 的原始文件为准。不要把开发集结果理解成独立泛化性能，也不要把 BOPTEST 仿真表现理解为真实楼宇部署保证。

原审批材料见 `reports/week8/graph_approval_pending.json`；用户批准后完成执行，最终状态和重新审核结果见 `reports/week8/graph_approved_verification.json`。
