# Week 5 统一模型加载约定

## 加载和物理尺度

统一入口为 `models.dynamics_base.load_dynamics(spec, root, device='cpu', hashes=None)`。`spec` 沿用 Week 5 配置中的模型条目；`hashes` 传入配置顶层的 `sha256` 映射，逐文件检查检查点。条目中的 `sha256` 优先。未提供哈希时仅加载，不宣称已核验版本。

支持 persistence、linear、action_mlp、ib_mlp、ib_history_mlp、ib_gru。热工 MLP 检查点自行区分直接状态和残差预测。所有入口返回 NumPy 数组；神经网络在内部迁移到指定 PyTorch 设备，线性与 Persistence 在 CPU 上计算。不可用设备由 PyTorch 明确报错，不静默切换。

热工状态顺序为室温、墙温，单位 °C；动作是加热功率 kW；扰动顺序为室外温度 °C、内部负荷 kW。输入单样本 `(2,)` 或批量 `(N,2)`，返回同形状下一状态。标准化与反标准化在模型内部完成，调用方不得重复标准化。

IB 使用公开观测的原始 benchmark 尺度，不能当作热工温度或 kW。观测顺序为 setpoint、velocity、gain、shift、fatigue、consumption，形状 `(6,)` 或 `(N,6)`；动作顺序 velocity、gain、shift，形状 `(3,)` 或 `(N,3)`，是适配器所定义的归一化增量命令。额外扰动传 `None`，setpoint 已包含在观测中。预测的是下一公开观测，不是不可见真实状态。

## 历史与重置

无记忆模型在每个 episode / 窗口开始调用 `reset()`。热工模型不接受历史参数。

IB 历史模型先调用 `reset(observation_history=..., action_history=...)`，传入当前时刻之前的 19 个观测及各自的历史动作，分别是 `(19,6)`、`(19,3)`，也可带批量维。随后调用 `predict_next(current_state, current_action, None)`，组成共 20 步输入。不得用未来观测补齐历史；历史不足的起点跳过。当前观测 MLP 只使用当前观测及动作，不需要 19 步上下文。

也可直接调用 `predict_window(window)`，输入 `(20,9)` 或 `(N,20,9)`，前六列观测、后三列该步动作。每次 GRU 前向都省略隐藏状态参数，由 PyTorch 从零初始化；同一批中的窗口各自独立，跨窗口不传递隐藏状态。

`predict_next` 不会自动推进历史，以便 MPC 用同一历史反复评估候选动作。自由递推时，评测器应将自身预测的观测和实际选用的动作加入窗口，丢弃最旧一行，再显式重置历史。Teacher forcing 则加入已记录的真实观测。两者不能混用。空 `reset()` 清除历史；历史模型在未初始化时预测会报错。

## 核验

运行 `python scripts/check_week5_model_loading.py`。结果写入 `reports/week5/results/model_loading_checks.json`。

检查覆盖 8 个热工配置与 9 个 IB 检查点：热工单样本与批量一致、非法数值拒绝；IB 与原始网络同设备预测一致、A/B/A 调用互不影响、显式历史入口一致、空重置后必须重新初始化。旧预测文件与 CPU 加载结果的最大差异单独保存，不把跨设备文件逐位一致作为通过条件。后续正式评测须固定设备与精度并重新导出预测。

IB 加载支持不代表它已经进入热工主榜：两个环境的维度、单位及信息集不同，必须分别评测。
