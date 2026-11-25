# Week 4 动作条件转移数据规范

格式版本为 1.0。第一版适用于 SimpleThermalEnv，采集入口为 data.collect_trajectories.collect_episodes，返回 DataFrame。保留 Week 2 的标量列以兼容线性辨识，新增规范字段中的数组以 JSON 字符串表示。CSV 读取后使用 json.loads 恢复数组。

## 转移与时间对齐

每行表示一条 $(x_t,u_t,d_t,x_{t+1})$ 转移。先读取当前状态与观测，再计算动作，环境把动作应用于 $[t,t+\Delta t)$ 后产生下一状态。时间字段使用模拟小时，source_step 从 0 开始。最后一行保留最终转移，并通过 terminated、truncated 记录结束。后续窗口在 episode 内构造。

| 字段 | 顺序、单位与定义 |
|---|---|
| state、next_state | 两维 [T_air,T_wall]，°C，真实模拟状态 |
| observation、next_observation | 三维 [T_air,T_wall,T_out]，°C，包含测量噪声 |
| action_requested | 一维请求加热功率，kW |
| action_applied | 一维实际生效功率，kW，动力学训练使用此列 |
| exogenous | 两维 [T_out,Q_load]，单位 [°C,kW]，实际作用于本步的扰动 |
| exogenous_available_at_decision | [当前带噪室外温度观测,固定已知内部负荷]，单位同上 |
| process_noise | 两维实际状态增量噪声，°C，仅诊断和复现使用 |
| time、next_time、dt | 动作开始、结束及保持时长，hour |
| action_hold_steps | 每行命令在一步内保持，当前为 1；随机策略可能连续多行沿用同一功率 |
| cost、energy_kwh | 环境示例成本与本步能耗 |

当前环境对非法动作报错，成功采集的 requested 与 applied 相同。将来接安全守卫时，仍须分别记录请求和实际执行动作。

## 信息边界

热工主实验以完整真实状态建模，作为全状态系统辨识设定。策略只读取带噪 observation。实际 exogenous 是模拟器诊断真值，和决策时可见量分别保存。

使用实际扰动拟合或进行给定输入序列的离线 rollout 时，应将结果标注为 oracle-disturbance 条件。闭环规划采用 exogenous_available_at_decision 或由当时历史构造的预测。模型自由递推只在起点读取真实状态，之后输入模型自身预测。模型的未来动作可由规划候选给定，未来扰动的信息来源需随实验配置保存。

## 溯源和分组

| 字段 | 定义 |
|---|---|
| episode_id | 场景、种子与回合编号组成的标识 |
| scenario_id | 热工配置指纹与初始室温/墙温组成的场景标识 |
| config_id | 完整环境配置 JSON 的 SHA-256 前 16 位 |
| environment_config | 完整物理参数、噪声和时间配置 JSON |
| seed、environment_seed | 相同的环境种子，后者兼容旧列 |
| policy_seed | 初态采样及分段随机动作的随机源种子 |
| source_step、step | 同一条原始转移的零基步编号 |
| episode | 本次采集调用中的整数回合编号，兼容旧列 |
| policy | random 或 rule |
| terminated、truncated、end_reason | 环境结束、时间截断与原因 |

每行可由 (episode_id,source_step) 定位。合并重复采集数据前检查此键，避免同一转移重复进入不同切分。按 episode 切分可做同分布评测，按 scenario 切分可做场景泛化评测。两个切分均应保存 ID 清单。
