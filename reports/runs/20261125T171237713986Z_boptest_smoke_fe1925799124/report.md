# Benchmark run

Run: `20261125T171237713986Z_boptest_smoke_fe1925799124`

所有闭环动作均由 BOPTEST 服务执行；开环输入模式与天气信息集见 config.yaml。

## 开环指标（每个 episode）

| 模型 | Episode | 条件 | 模式 | Horizon | 室温 RMSE °C | 有效比例 |
| --- | --- | --- | --- | --- | --- | --- |
| residual_mlp_seed42 | id_test_day160 | clean | teacher_forcing | 1 | 0.0407387 | 1.000 |
| residual_mlp_seed42 | id_test_day160 | clean | teacher_forcing | 4 | 0.0284854 | 1.000 |
| residual_mlp_seed42 | id_test_day160 | clean | free_running | 1 | 0.0407387 | 1.000 |
| residual_mlp_seed42 | id_test_day160 | clean | free_running | 4 | 0.133863 | 1.000 |

完整六变量 MAE/RMSE、失败数与物理异常比例保存在 metrics.json。

## 真实闭环指标

| 任务 | 模型 | Episode | 条件 | 成本 | 能耗 kWh | fallback |
| --- | --- | --- | --- | --- | --- | --- |
| closed_loop | residual_mlp_seed42 | id_test_day160 | clean | 0.2738 | 0.00836962 | 8 |
| closed_loop | embedded | id_test_day160 | clean | 0.2738 | 0.00836962 | 0 |

原生 KPI 的 final 区间包含历史上下文；只对可加的累积量作差，不对峰值或比率作差。

排序评测条目：0。各回合独立保留，未将时间步视为独立重复。

status.json 的 completed 表示流程完成；有效比例与失败事件需同时查看。
