# Week 5：统一世界模型评测

## 当前完成范围

Week 5 主实验任务 1–8 已完成：冻结协议、统一模型加载、单步与 80 步 rollout、有效性诊断、episode/模型种子两层统计及配对差值、图表和一键复现。IB 历史模型与输出投影属于独立扩展；概率校准按用户指示尚未执行。本页预测指标不能直接推出控制收益。

## 单步实验设置

沿用 `configs/experiments/week5_evaluation.yaml`，运行前逐项核验其中所有数据与检查点 SHA-256。18 个隔离测试 episode，每回合取原始步 0–64，共 1170 个共同起点；这些起点同时支持后续完整 80 步预测。单步实验也使用相同清单，不另取全部 144 步。

每次从真实当前室温与墙温预测下一状态；输入当前已生效加热动作、当前可得的带噪室外观测和已知内部负荷。目标为下一步真实状态，误差单位 °C。起点处两种天气条件相同，teacher forcing 与 free running 的第一步也相同，因此只计算一份单步结果。

模型为 Persistence、已训练线性模型，以及种子 42、43、44 的直接 MLP 和残差 MLP，共 8 个配置。不在测试集拟合标准化参数、训练或选参。每个起点调用 `reset()`。IB 数据与模型不进入热工表，避免混合单位和信息集。

## 汇总规则与工况

先计算每个 episode 的各变量 MAE/RMSE，再对 episode 等权平均。`metrics_by_condition.csv` 中的 `*_episode_sd` 是 episode 间样本标准差；`metrics_across_seeds.csv` 中的 `*_seed_sd` 是各训练种子总体指标间的样本标准差。两者不可互换，不能把多个种子的重叠窗口当作独立样本。

| 分组 | 固定定义 |
|---|---|
| all | 所有共同起点 |
| policy | 轨迹记录的采集策略来源 |
| initial_air | 回合第 0 步真实室温：小于 19、[19,21)、不低于 21 °C |
| action | 当前已生效功率：小于 1、[1,4)、不低于 4 kW |
| policy_initial_air | 策略来源与回合初温区间的组合 |

初温仅用于事后分组，不增加模型信息。同一个 episode 可在不同动作组出现；每组单独按 episode 等权汇总，不把各组均值再平均成总体。没有样本的分组不生成虚构指标，组内只有一个 episode 时标准差留空。

推理异常、非法形状或 NaN/Inf 保留 `failure_reason`，该次预测两个状态均记为缺失；误差只用有效预测计算，同时报告 requested、valid、failed、valid_fraction 及有有效误差的 episode 数。整个 episode 全失败时仍保留该 episode，但其误差为空。本轮 9360 次模型预测全部有效，对应 18720 条逐变量输出记录。

## 单步结果

以下均值先经过 episode 等权汇总；MLP 的 ± 为三个训练种子之间的样本标准差。确定性基线只运行一个模型，不虚构种子离散性。每个模型的 episode 标准差另见 CSV。

| 模型 | 变量 | MAE（°C） | RMSE（°C） |
|---|---|---:|---:|
| Persistence | 室温 | 0.130761 | 0.146759 |
| Persistence | 墙温 | 0.016650 | 0.019392 |
| Linear | 室温 | 0.007985 | 0.010068 |
| Linear | 墙温 | 0.008342 | 0.010377 |
| 直接 MLP | 室温 | 0.009172 ± 0.000674 | 0.011672 ± 0.000960 |
| 直接 MLP | 墙温 | 0.009626 ± 0.000476 | 0.012042 ± 0.000641 |
| 残差 MLP | 室温 | 0.008411 ± 0.000073 | 0.010623 ± 0.000001 |
| 残差 MLP | 墙温 | 0.008371 ± 0.000012 | 0.010412 ± 0.000018 |

当前系统本身为线性热工动力学加噪声，线性模型与结构匹配，因此它的单步误差最低并不意外。残差 MLP 接近线性模型且本轮种子间差异更小。Persistence 的墙温误差比室温小，反映墙体在 10 分钟内变化较慢，不能因此认为它具备正确的动作响应。是否长期更优仍须看后续自由递推。

## 文件与复现

在项目根目录激活 mvp 后运行：

```bash
python scripts/evaluate_week5.py --config configs/experiments/week5_evaluation.yaml --device cpu
python -m unittest tests.test_week5_prediction tests.test_metrics
```

结果目录为 `reports/week5/results/one_step/`：

| 文件 | 内容 |
|---|---|
| predictions.csv.gz | 每个模型、episode、原始步、变量的真值/预测、失败原因、输入、工况及检查点哈希 |
| metrics_by_episode.csv | 每个 episode 在各工况中的指标与有效样本数 |
| metrics_by_condition.csv | 每个模型配置的工况均值、episode 样本标准差与计数 |
| metrics_across_seeds.csv | 模型族的训练种子均值与种子样本标准差 |
| model_manifest.json | 加载配置、检查点哈希、请求设备 |
| protocol_snapshot.json | 本次冻结配置、起点数量、设备、工况阈值及汇总口径 |

统计检查覆盖：不等长 episode 的等权汇总、失败 episode 保留、种子与 episode 离散性分离、逐变量指标，以及已有指标边界测试，共 6 项通过。

## 长时序 rollout

沿用相同 1170 个起点，每个预测 80 步。8 个模型配置分别执行 teacher_forcing / free_running 与 recorded_observation / persistence_forecast 的组合，共 37440 条条件预测轨迹。本轮无推理失败或超过预设数值发散边界的轨迹；这不代表所有物理合理性诊断均已完成。

Teacher forcing 在第 k 步输入测试轨迹真实当前状态；其长期曲线是不同未来时刻的单步误差，不是自主推演误差。Free running 仅在起点读一次真实状态，后续以自己的预测状态递推。两者均输入记录的生效动作序列，不模拟控制器重新决策。

Recorded observation 条件使用未来各步记录的带噪室外观测与已知内部负荷，是离线条件预测。Persistence forecast 条件将起点室外观测保持 80 步，内部负荷依然使用已知负荷。两类结果分别保存，不能把离线给定天气描述成提前知道真实未来天气。两种模式和天气条件的第 1 步均已核对与单步评测一致。

对 1–80 每一步均输出以下三种口径，并额外保存关键步长 1、5、10、20、40、80 的表：

| definition | 含义 | 失败时的口径 |
|---|---|---|
| terminal | 仅第 H 步的 MAE/RMSE | 只用第 H 步有效的起点 |
| prefix_complete | 第 1–H 步全部误差的 MAE/RMSE | 只用完整 H 步有效的起点 |
| prefix_available | 第 1–H 步可用前缀误差的 MAE/RMSE | 保留提前停止前的可用点，同时报告完整轨迹数量 |

前缀 RMSE 为全部选中时间点平方误差均值的平方根，不是各步 RMSE 的算术平均。每种口径先在 episode 内算，再对 episode 等权汇总。表中包含 requested、valid_origins、complete_origins、valid_points、valid_fraction；没有有效误差的 episode 数量也可追溯。发散阈值触发点本身若有限则保留，随后步骤缺失，避免掩盖导致停止的大误差。推理异常不写入伪造状态。

下面展示 recorded_observation + free_running 的室温末端 RMSE（°C）；MLP 的 ± 为训练种子样本标准差。

| 模型 | 1 步 | 20 步 | 80 步 |
|---|---:|---:|---:|
| Persistence | 0.146759 | 0.863315 | 1.064952 |
| Linear | 0.010068 | 0.034518 | 0.051294 |
| 直接 MLP | 0.011672 ± 0.000960 | 0.053088 ± 0.007372 | 0.084034 ± 0.020827 |
| 残差 MLP | 0.010623 ± 0.000001 | 0.042085 ± 0.001175 | 0.056719 ± 0.003510 |

当前示例中误差随自由递推增长，残差 MLP 长期更接近线性模型。该表仅展示一种扰动条件和室温变量；完整墙温、teacher forcing、天气保持条件及前缀指标均在 CSV 中，不能据此概括其他条件。

运行与检查：

```bash
python scripts/evaluate_week5.py --stage rollout --device cpu
python scripts/evaluate_week5.py --stage all --device cpu
python -m unittest tests.test_week5_rollout tests.test_week5_prediction tests.test_metrics
python scripts/check_week5_rollout.py
```

`reports/week5/results/rollout/` 保存：

| 文件 | 内容 |
|---|---|
| MODEL_predictions.npz | actual 及四种模式/天气组合；数组轴为起点、1–80 步、室温/墙温 |
| origins.csv | 数组起点轴对应 episode、原始 source_step、策略及回合初温 |
| metrics_by_episode.csv | 逐 episode、逐 horizon、逐变量的三种误差口径 |
| metrics_by_horizon.csv | 每个模型的 episode 等权均值和样本标准差，覆盖 1–80 步 |
| metrics_key_horizons.csv | 固定关键 horizon 子表 |
| metrics_across_seeds.csv | 模型族的训练种子汇总 |
| failures.csv | 所有请求轨迹的第一失败步与原因；failure_step=0 表示未失败 |
| protocol_snapshot.json | 冻结配置与哈希、设备、输出轴、信息集和停止规则 |

新增测试检查未来状态隔离、天气条件隔离、发散后停止与前缀有效计数；与前一步测试合计 10 项通过。另核对全部 8 个模型的首步与既有单步输出一致。已有 Week 4 `evaluate_rollouts` 入口保留，避免改变旧实验的默认时域与产出。

## Rollout 指标图

运行 `python scripts/plot_week5.py`，输出 `reports/week5/figures/`。所有图同时保存 PNG、PDF、SVG；关键步长的完整 MAE/RMSE 表为 `key_horizon_metrics.csv`。

![末端误差](figures/terminal_rmse.png)

上排给定未来各步室外观测，下排将室外温度保持起点值；左列室温、右列墙温。纵轴为对数刻度，越低越好。MLP 阴影为三个训练种子间 ±1 样本标准差，不是置信区间或 90% 预测区间。给定天气时线性与残差 MLP 更优；天气保持条件下误差明显增大，模型排名也可能改变，不能仅用上排排名概括整体。

![前缀误差](figures/prefix_rmse.png)

前缀图评估第 1–H 步整体误差；末端图只评估第 H 步。两者均先按 episode 计算，再等权汇总。

![真实状态输入与自由递推对照](figures/teacher_vs_free.png)

实线为自由递推，虚线为每步输入真实当前状态。实虚线间距展示状态递推误差积累；天气保持条件下虚线仍可能升高，因为未来天气估计也有误差。

![固定起点预测轨迹](figures/rollout_example.png)

固定第一个测试 episode 的 source_step=0，没有按误差挑选样例。黑线为模拟器真实状态；模型均以记录动作作为输入。该图是单个示例，整体优劣应依据 18 个 episode 的指标。具体 episode ID 及数据处理记录见 `figure_notes.json`，轨迹数值见 `example_source_data.csv`。

已检查最终面板对齐、PDF 文字大小与文字碰撞；当前为报告展示尺寸及 300 dpi 预览，不宣称符合特定期刊投稿尺寸。

## 预测有效性检查

实现位于 `metrics/validity.py`，由 `evaluators/rollout.py` 在线调用；它不裁剪或修正预测，也不改变冻结阈值与模型。参数来自 Week 5 YAML；速率方程参数来自每个 episode 记录的 `environment_config`。

| 事件 | 定义 | 行为 |
|---|---|---|
| nan / inf | 模型原始输出包含 NaN / 正负 Inf，逐变量记录 | 停止该起点；失败步不进入误差，保留此前预测 |
| inference_error | 推理抛异常或输出形状不合法 | 停止并保留异常类型及文本 |
| observation_bounds | 超出环境声明的观测边界 | 当前热工观测空间无有限温度界，此项不适用，计数 0 不能解释为通过有限安全边界 |
| plausibility_bounds | 温度超出 [-10,50] °C | 记录越界幅度，继续 |
| rate | 预测增量超出热工方程推导的输入范围包络 | 记录上下界及超出幅度，继续 |
| divergence | 温度超出 [-100,100] °C | 保留有限的触发点，停止后续步骤 |

速率包络使用室外 [-5,17] °C、功率 [0,5] kW、内部负荷 0.3 kW，空气/墙体热容和热交换系数取环境记录，每步增量额外加减 0.03 °C。计算以当前输入状态为基准：自由递推用模型自身状态，teacher forcing 用当前真实状态。只用 1e-12 数值容差避免浮点边界误报。该包络是实验启发式检查；过程噪声无界，因此不是严格物理可行域或安全保证。

`first_anomaly_step` 是首次出现任一诊断事件的 horizon；`first_failure_step` 只记会停止轨迹的事件。均从 1 开始，0 表示在本次时域内没有对应事件。异常事件与导致停止的失败不能混称。

### 本轮检查结果

共 37440 条条件轨迹、2995200 个预测状态向量。没有 NaN、Inf、推理异常、温度合理性越界或数值发散。直接 MLP 的种子 43 检测到以下速率异常，其余模型配置没有触发：

| 模式 | 天气输入 | 受标记起点 / 请求起点 | 异常预测步 / 请求预测步 |
|---|---|---:|---:|
| free_running | 起点天气保持 | 12 / 1170 | 63 / 93600 |
| free_running | 各步记录观测 | 12 / 1170 | 59 / 93600 |
| teacher_forcing | 起点天气保持 | 12 / 1170 | 82 / 93600 |
| teacher_forcing | 各步记录观测 | 12 / 1170 | 72 / 93600 |

这里每个起点每一步只计一次，即使两个变量同时异常也不重复计步；逐变量事件另存。四种条件合计 48 条受标记轨迹、276 个异常步，不能把它们当作 48 个独立实验：不同模式、条件共享起点，且起点窗口也重叠。速率异常只标记继续，因此本轮长期误差指标保持原样；它说明部分输出超过预设包络，需要结合具体状态与幅度分析，不能直接称为模型发散。

### 可追溯输出与复现

`reports/week5/results/rollout/` 新增：

| 文件 | 内容 |
|---|---|
| validity_events.csv.gz | 仅实际触发的逐变量事件：模型、模式、天气条件、episode、原始起点、horizon、输入状态、原预测、阈值、超出幅度及停止标志 |
| validity_by_origin.csv | 全部请求轨迹；各类异常步数、首次异常/失败步、请求/实际尝试/有限输出步数 |
| validity_summary.csv | 每个模型、模式和天气条件的计数；失败起点比例、有限输出占全部请求步的比例 |
| MODEL_predictions.npz | 原有误差计算用数组保留；每个组合新增 `__raw` 和 `__attempted` 数组 |

`__raw` 保存未修正输出，包括 NaN/Inf；`__attempted` 为布尔掩码。这样可以区分失败步输出 NaN 与停止后的未执行空位。非法形状或完全没有返回输出的异常通过事件文本追踪，不伪装成模型产生 NaN。失败前的轨迹仍在原始数组中。

```bash
python scripts/evaluate_week5.py --stage rollout --device cpu
python -m unittest tests.test_validity tests.test_week5_rollout tests.test_week5_prediction tests.test_metrics tests.test_dynamics_api
python scripts/check_week5_rollout.py
```

15 项测试通过；NaN、Inf、非法形状、推理异常、温度越界、速率边界和发散停止均有人工构造检查，原始前缀与首次失败步验证通过。全部 8 个模型的 rollout 首步仍与单步评测一致。

## 有效性与统计验收补齐

本节对应有效性验收建议的后续补齐。热工主协议、原数据及检查点均未改变；IB 与投影比较使用独立扩展配置。不得把独立扩展描述为原主协议在测试前已包含的内容。

### 逐 horizon 与训练种子表

主评测和独立扩展的 `metrics_by_horizon.csv` 现在同时提供：

| 字段 | 定义 |
|---|---|
| requested | 该模型、模式、条件请求的全部起点数量 |
| valid_origins / valid_fraction | 对当前误差口径具有有限可用输出的起点数及比例 |
| failed_origins / failure_fraction | 截至 H 步已经触发停止失败的起点数及比例 |
| unavailable_origins | 对当前误差口径没有可用输出的起点数，即 requested−valid_origins |
| complete_origins | 前 H 步全部有有限输出的起点数量 |
| valid_points | 实际进入当前指标的预测时间点数量 |
| episodes / episodes_with_valid_error | 请求 episode 数量及具有有效误差的 episode 数量 |

“失败”和“缺失”不是同一件事。发散触发步若仍有限，该步保留在末端误差中，但该轨迹已经记为失败；只有后续步缺失。可用前缀指标也可能仍使用一条已失败轨迹的早期预测。以上计数分别保留，不互相代替。

`metrics_across_seeds.csv` 补充 `runs_with_valid_error`、`requested_per_run`、`requested_model_origins`、`failed_model_origins`、`unavailable_model_origins`、有效率/失败率的种子均值与样本标准差。`*_model_origins` 是模型运行×起点的计数，不是独立实验样本数。单步种子表也补充实际运行数、请求数、失败数和有效率。Persistence 与线性只报告一次运行，种子 SD 留空。

有效率在这里明确指有限数值预测可用率，不等于物理合理率、声明空间合规率或安全性。异常事件表与计数必须一起阅读。热工 `validity_summary.csv` 明确写入 `observation_bounds_applicable=False`，零次声明空间越界不作为安全性结论。

### 独立输出投影对照

配置：`configs/experiments/week5_projection.yaml`。保留原 8 个配置，额外加入三个训练种子的 `projected_mlp`，共 11 个配置。投影使用原训练检查点、相同起点、动作与天气条件；修正由已有 `RateConstrainedDynamics` 执行，参数固定为室外 [-5,17] °C、功率 [0,5] kW、0.03 °C 噪声余量。它使用额外热工物理先验，身份与原 MLP 分开，不将其当作同一原模型。

输出目录：`reports/week5/results/projection/rollout/`。`projection_summary.csv` 提供触发步数、分量数及最大修正；每个投影模型额外的 `*_projection_diagnostics.npz` 保存 before / after / triggered 数组。其数组起点对应同目录 `origins.csv`。这些 before 是投影模型在自身递推状态上计算的基础网络输出，不应误当成原始未约束模型的整条自主轨迹；原始模型完整轨迹另外保留。

本轮 seed43 在四种模式/天气组合中分别修正 72、59、82、64 步，最大单分量修正约 0.031764 °C。seed42、44 没有触发投影。复算与原记录有不超过 5.4e-7 °C 的浮点差异，`max_replay_difference_c` 单独记录，不计入投影修正量。

给定未来各步天气观测的自由递推室温末端 RMSE（°C），三个训练种子的均值：

| 模型 | H=20 | H=80 |
|---|---:|---:|
| 原始 MLP | 0.053088 | 0.084034 |
| 输出投影 MLP | 0.053207 | 0.084118 |

投影减少了超包络输出，但本轮精度略差，不能宣称约束必然提高预测准确率。投影轨迹因先前修正而改变，所以之后的触发步数可以与原始模型异常步数不同。

### IB GRU 独立评测

配置：`configs/experiments/week5_ib.yaml`，固定原轨迹及所有检查点哈希，沿用原训练/验证/测试 episode 划分。测试 5 个完整 episode；每回合 source_step=19–220，共 202 个起点、总计 1010 个共同起点；每个起点最长 80 个仿真 step，不将 IB step 解释成热工的 10 分钟。

比较 IB Persistence，以及当前观测 MLP、20 步历史 MLP、GRU 的 42/43/44 三个种子，共 10 个配置。当前观测组与 history20 信息组在配置中明确分开；历史 MLP 和 GRU 使用相同历史信息，二者更适合公平比较。当前观测模型与历史模型差异同时包含信息量差异，不能全部归因于网络结构。

所有模型预测六维公开观测，不假装观测等于完整隐藏状态。历史输入为 20 个按时序排列的观测及各自出发动作；未来动作使用记录序列。自由递推仅使用起点之前真实历史，随后逐步加入模型预测观测与对应动作；teacher forcing 才读取后续真实当前历史。GRU 每次前向隐藏状态从零开始，窗口、候选和 episode 间不传递隐藏状态。所有模式的首步已核对一致。

结果目录：`reports/week5/results/ib_history/`。误差字段使用 `mae_native` / `rmse_native`，单位为 IB 原生 benchmark 尺度，不是摄氏度；各变量分别报告，不能跨变量直接平均，也不进入热工主榜。

示例：consumption 的自由递推末端 RMSE，均值 ± 训练种子样本标准差：

| 模型 | H=1 | H=20 | H=80 |
|---|---:|---:|---:|
| IB Persistence | 13.897 | 117.777 | 137.662 |
| 当前观测 MLP | 9.688 ± 0.147 | 35.234 ± 9.918 | 117.494 ± 131.435 |
| 20 步历史 MLP | 4.772 ± 0.081 | 33.008 ± 9.520 | 47.254 ± 8.547 |
| GRU | 4.562 ± 0.039 | 18.783 ± 1.627 | 24.180 ± 4.261 |

以上只代表 consumption 一个变量，不能代替其他五个变量的结论。当前观测 MLP 的 80 步种子间差异很大，均值需要和逐种子结果一起阅读。

20200 条“模型×模式×起点”预测轨迹没有推理停止失败，但九个神经模型均出现声明空间越界。严格按适配器声明的六维范围检查：[0,100] 对应 setpoint/velocity/gain/shift，[0,1000] 对应 fatigue/consumption。保留原输出、越界变量和幅度，标记继续，不做自动裁剪。因此不能将有限预测比例 100% 写成“全部预测符合约束”。详见 `validity_events.csv.gz`、`validity_by_origin.csv`、`validity_summary.csv`。

IB 数值发散使用独立协议固定的绝对值 1e6 阈值，它是数值启发式阈值，不是物理范围。未建立可信的物理变化速率边界，该检查明确记为不适用；不套用热工温度或变化速率阈值。

### 可选 episode 分组置信区间

三个结果目录均新增 `episode_bootstrap_ci.csv`：对关键 H=1、5、10、20、40、80 的 terminal / prefix_complete MAE、RMSE，采用 2000 次 episode 分组重采样、固定种子 2026，给出百分位 95% 区间。

每次抽样保留整个 episode 内所有重叠窗口；不重采样单个窗口。训练种子集合固定，先计算回合内的种子均值，不把种子当作额外独立回合。所有模型使用相同 episode 抽样索引，便于后续配对分析。当前文件是各模型指标区间，并不是模型差值的显著性检验。若某个 episode 或模型运行没有有效误差，该组不生成误导性的区间，留空并记录原因。

热工独立评测单位为 18 个 episode，IB 只有 5 个，后者区间稳定性有限。它们描述当前固定训练种子集合下的 episode 变异，不包含重新训练导致的全部不确定性。原冻结协议 v1 主表仍报告均值/SD，本文件作为明确标注的补充统计，不能追溯声称其为原协议预先承诺的区间。

### 复现与验证

首次扩展配置已生成并冻结，常规复现无需重新运行 freeze 脚本。项目根目录执行：

```bash
python scripts/evaluate_week5.py --stage all --device cpu
python scripts/evaluate_week5.py --stage rollout --config configs/experiments/week5_projection.yaml --device cpu
python scripts/check_week5_projection.py
python scripts/evaluate_week5_ib.py
python scripts/check_week5_extensions.py
python scripts/bootstrap_week5.py
python scripts/bootstrap_week5.py --folder reports/week5/results/projection/rollout
python scripts/bootstrap_week5.py --folder reports/week5/results/ib_history
```

18 项统计、故障注入、窗口隔离、投影及回归测试通过；已核对主评测首步与单步输出一致，IB 全部模型两种模式的首步一致、每模型形状为 (1010,80,6)。投影修正前后输出已复算核验。原模型产出仍保留，所有新增比较均有独立目录与配置。

## 任务 6：按独立单位汇总与 episode 配对

新增 `evaluators/summary.py`，单步、热工 rollout 与 IB 评测复用同一套 episode / 模型种子统计；原 `summarize_rollout_metrics` 保留兼容入口。原始窗口误差先在 episode 内聚合，统计模块只接收逐 episode 指标，不将重叠窗口当作独立重复。

主统计顺序为：每个模型种子先对 episode 等权计算均值与 episode 样本 SD，再对固定训练种子计算均值与种子样本 SD。确定性基线仅一次模型运行，种子 SD 留空。用于模型族配对时，先在同一 episode 内平均固定模型种子，再对相同 episode 做差；完整数据下其平均与主统计均值一致，但配对差的 SD 专门反映 episode 间的差异。

新增检查拒绝重复 episode 行及静默缺失的模型回合行。失败 episode 必须显式保留为有计数、误差可为空的一行，不能通过删除整行绕过分母。

### 配对定义和失败支持

配对键固定为 episode、mode、condition、horizon、definition、variable、unit。不同环境和单位不混合；模型族名称按字母顺序组成 A、B，差值定义为：

$$
\Delta_e=\overline{\mathrm{error}}_{A,e}-\overline{\mathrm{error}}_{B,e}
$$

负值表示 A 的误差更小，正值表示 B 的误差更小。先保存每个 episode 的差值，再计算这些差值的回合均值和样本 SD；不以两个模型各自标准差相减替代配对差标准差。

主配对差使用保守的完整支持规则：两边相同 episode 的请求起点数一致、所有模型种子均有有限误差、前 H 步完整预测、且截至 H 无停止失败。若不满足，仍保留这一配对行，`paired_valid=False`、`pair_status` 写明缺失或支持不完整，主误差差值留空。配对汇总同时报告 requested_episode_pairs、valid_episode_pairs、invalid_episode_pairs、paired_valid_fraction，不将失败配对隐藏。

原主表里的有效样本误差继续按原协议计算；主配对差更严格，因此在将来有失败模型时二者可使用不同支持。`delta_max_run_failure_fraction` 是两边最差种子失败比例之差，字段明确标注 max_run，不将它误称为平均失败比例。

IB 配对还记录 `information_a`、`information_b`、`same_information_set`：历史 MLP 与 GRU 都属于 history20；当前 MLP 与 Persistence 属于 current。跨信息组的配对保留为诊断，不能把差异全部归因于模型结构。热工投影配置虽使用相同状态/动作输入，但额外利用热工先验，比较时仍应说明此前已记录的方法差异。

### 产出与实测示例

以下三个目录均补齐相同统计产出：

- `reports/week5/results/rollout/`
- `reports/week5/results/projection/rollout/`
- `reports/week5/results/ib_history/`

| 文件 | 内容 |
|---|---|
| metrics_by_horizon.csv | 每个模型种子的 episode 等权均值、episode 样本 SD 和失败计数 |
| metrics_across_seeds.csv | 各模型族的训练种子均值、样本 SD、实际运行数和有效性计数 |
| metrics_by_episode_across_seeds.csv | 每个 episode 内的模型种子均值/SD，以及模型种子完整支持标记 |
| paired_episode_differences.csv.gz | 同一 episode 的模型族 A/B 指标、差值、支持状态及信息集标记 |
| paired_summary.csv | 全部 1–80 horizon 的 episode 配对差均值、样本 SD、请求/有效/无效配对数 |
| paired_key_horizons.csv | 关键 H=1、5、10、20、40、80 的配对子表 |
| summary_protocol.json | 独立单位、统计顺序、配对方向、失败支持及信息集说明 |

本轮主实验保存 207360 行逐 episode 配对，投影扩展 345600 行，IB 独立扩展 86400 行。每行对应一个评测条件和回合，行数不等于独立 episode 数。

例如主实验 recorded_observation + free_running + H=80 + terminal + air：MLP−Linear 的 RMSE 差为 0.032740 °C，配对 episode 样本 SD 为 0.028210 °C，18/18 个 episode 配对有效。它表示该固定条件下 MLP 平均误差更大，不是显著性检验，也不能推断到其他信息条件。

### 复现和验收

无需重新训练或推理，即可从现有逐 episode 指标重建所有汇总与配对文件：

```bash
python scripts/summarize_week5.py
# 也可只处理主实验：
python scripts/summarize_week5.py --folder reports/week5/results/rollout
```

常规 rollout 和 IB 评测结束时也自动生成这些文件。单步评测同样复用 `summary.py` 的两层统计核心，保留既有工况拆分字段。

24 项测试通过，覆盖两层均值/SD、确定性基线单次运行、配对方向、失败配对保留、缺失/重复 episode 拒绝、不同起点支持不配对，以及此前的有效性/信息隔离/投影回归。新旧误差聚合已逐项核对一致。本次没有执行上一条被取消的概率校准任务。

## 任务 7：表格与图

`scripts/plot_week5.py` 已扩展，常规运行从已完成的主评测结果读取数据，不重新训练或推理。这里展示冻结热工主实验四个模型族，IB 原生尺度与投影模型仍保存在独立扩展中，不混入这些摄氏度主榜。

| 图 | 文件名 | 阅读方式 |
|---|---|---|
| 末端误差—horizon | terminal_rmse.png | 第 H 步 RMSE；左右为室温/墙温，上下为给定天气观测/保持起点天气 |
| 前缀误差—horizon | prefix_rmse.png | 第 1–H 步整体 RMSE，先回合汇总再种子均值 |
| 两种递推模式 | teacher_vs_free.png | 实线自由递推，虚线每步输入真实状态 |
| 同起点轨迹 | rollout_example.png | 固定首个测试起点，真实状态与模型预测对照 |
| 有限预测比例 | valid_prediction_fraction.png | 每个模型在每个 horizon 的有限预测可用率，固定 0–100% 色阶 |
| 策略来源工况分布 | condition_distribution_policy.png | H=80 的回合末端 RMSE，按 mpc/random/rule 采集来源分组 |
| 初始室温工况分布 | condition_distribution_initial_temperature.png | H=80 回合误差，按 <19、[19,21)、>=21 °C 分组 |
| 排序随 horizon 变化 | ranking_by_horizon.png | 按种子均值 RMSE 排名，1 为最低误差；不同天气条件分别排名 |

![有限预测可用比例](figures/valid_prediction_fraction.png)

本轮主实验所有 horizon 的有限预测可用率为 100%，不代表没有变化速率异常。有效率以全部请求起点为分母，不能解释成物理合规率或区间覆盖率。

![按采集策略的回合误差分布](figures/condition_distribution_policy.png)

每个点是一个 episode 在固定训练种子平均后的误差；箱体为 Q1/中位数/Q3，须线为最小值到最大值。三个采集策略各 6 个测试回合。所有回合都显示，没有把大误差回合当作离群点删除。采集策略只说明测试轨迹来源，此图不是规则/MPC 闭环控制收益对比，也不能从分组关联推出策略的因果影响。

![按初始温度的回合误差分布](figures/condition_distribution_initial_temperature.png)

初温组分别有 6、1、11 个回合；中间组只有一个回合，所以图上只出现点/线，不能据此判断该温区的稳定性或总体分布。初温为回合真实起始室温，仅用于事后分组，不额外提供给模型。

![排序随预测时长变化](figures/ranking_by_horizon.png)

给定天气观测时，本轮主实验四个模型族的 RMSE 排序在 H=1–80 保持稳定；起点天气保持条件下，排序随时长变化。它反映天气输入误差与模型误差共同作用，不能把某个模型排名上升直接理解成其动力学更准确。排名也不表示误差差异具有统计显著性。

排序只对完整数值支持、无停止失败的模型族生成，失败或支持不完整时不以幸存子样本强行排名；具体 eligibility 与数值均保留在表中。误差曲线阴影是训练种子 SD，工况箱体分布则是 episode 变异，两者含义不同。误差图纵轴采用对数刻度，越低越好。

### 表格与源数据

全部位于 `reports/week5/figures/`，图像也保存同名 PDF 与 SVG：

- `evaluation_tables.md` / `evaluation_tables.csv`：关键 H=1、5、10、20、40、80 的 MAE/RMSE、训练种子 SD、有限有效率与排序，天气条件和变量分别列出。
- `key_horizon_metrics.csv`：完整关键步长表，包含两种模式和所有误差定义。
- `model_ranking_by_horizon.csv`：全部 1280 个模型族×天气条件×变量×horizon 的原始排名数据和可排名标记。
- `condition_distribution_source.csv`：分布图每个 episode 的真实源指标及工况元数据。
- `ranking_and_validity_source.csv`：排序及有限可用率绘图输入。
- `diagnostic_figure_notes.json`：统计单位、范围、样本量及绘图口径。

复现：

```bash
python scripts/plot_week5.py
```

已核对排序表覆盖 4 个模型族×2 种天气条件×2 个变量×80 horizon；每个工况分布面板完整保留 18 个 episode。最终导出通过面板对齐、PDF 文字大小及文字碰撞检查，并逐图查看。报告图使用 300 dpi 预览和可编辑矢量，不宣称符合特定期刊的投稿尺寸。

<!-- WEEK5_REPRODUCTION_START -->

## 一键复现：最近一次验收

- 配置：`configs/experiments/week5_evaluation.yaml`；SHA-256：`94bc23d553ec88320055b84bb262674494dee735795527dc973140af0ce82e64`。
- 运行：`20261124_230352_341929`；设备 `cpu`；Python `3.11.18`。
- 验收：18 个 episode、1170 个共同起点；4 个模型族、8 个模型配置，覆盖 [1, 5, 10, 20, 40, 80]。
- 逐步日志与数组由 `rollout/origins.csv` 映射到 episode_id 和 source_step；数组第二轴是预测步长，第三轴是 air/wall。
- 下面是给定天气观测的自由递推室温/墙温末端 RMSE（°C）；± 为训练种子 SD。其他条件、MAE 和前缀指标见完整表。

| 模型族 | 变量 | H=1 | H=5 | H=10 | H=20 | H=40 | H=80 |
|---|---|---:|---:|---:|---:|---:|---:|
| action_mlp | air | 0.01167 ± 0.00096 | 0.02906 ± 0.00504 | 0.04088 ± 0.00766 | 0.05309 ± 0.00737 | 0.07079 ± 0.01002 | 0.08403 ± 0.02083 |
| action_mlp | wall | 0.01204 ± 0.00064 | 0.03194 ± 0.00365 | 0.04701 ± 0.00675 | 0.06604 ± 0.01064 | 0.09380 ± 0.01954 | 0.12515 ± 0.02127 |
| linear | air | 0.01007 | 0.02082 | 0.02704 | 0.03452 | 0.04620 | 0.05129 |
| linear | wall | 0.01038 | 0.02226 | 0.02964 | 0.03759 | 0.04776 | 0.07484 |
| persistence | air | 0.14676 | 0.52982 | 0.64491 | 0.86331 | 0.97787 | 1.06495 |
| persistence | wall | 0.01939 | 0.08109 | 0.15269 | 0.27989 | 0.46884 | 0.66935 |
| residual_mlp | air | 0.01062 ± 0.00000 | 0.02349 ± 0.00007 | 0.03200 ± 0.00048 | 0.04208 ± 0.00118 | 0.05410 ± 0.00305 | 0.05672 ± 0.00351 |
| residual_mlp | wall | 0.01041 ± 0.00002 | 0.02251 ± 0.00013 | 0.02995 ± 0.00044 | 0.03869 ± 0.00073 | 0.05158 ± 0.00112 | 0.07970 ± 0.00358 |

[本次运行清单](results/reproduction/20261124_230352_341929/run_manifest.json) · [验收结果](results/reproduction/20261124_230352_341929/verification.json) · [测试日志](results/reproduction/20261124_230352_341929/tests.log)

[完整表格](figures/evaluation_tables.md) · [误差曲线](figures/terminal_rmse.png) · [有效比例](figures/valid_prediction_fraction.png) · [排序变化](figures/ranking_by_horizon.png)

```bash
conda activate mvp
bash scripts/reproduce_week5.sh
```

此一键命令复现冻结的热工主实验，不重新训练或重新冻结数据。IB、投影对照和补充分组 bootstrap 沿用各自独立脚本，避免混入主协议。

<!-- WEEK5_REPRODUCTION_END -->


## 任务 8：一键评测、报告与运行追溯

主配置仍为 `configs/experiments/week5_evaluation.yaml`。它固定输入资产哈希、完整 episode 划分、模型检查点、种子、信息集、horizon、有效性阈值和输出位置。一键复现不调用训练或 freeze 脚本；发现资产 SHA-256 不符时停止，不偷偷更新锁定值。

在项目根目录执行：

```bash
conda activate mvp
bash scripts/reproduce_week5.sh
```

等价 Python 命令：

```bash
python scripts/evaluate_week5.py --config configs/experiments/week5_evaluation.yaml --device cpu --stage reproduce
```

Shell 脚本第一个可选参数为配置路径，第二个为设备；默认 CPU。Python 使用已激活环境，也可通过 `WEEK5_PYTHON=/home/hera/anaconda3/envs/mvp/bin/python` 明确指定。CPU 为本轮已验收的默认运行设备；指定其他设备会记录设备差异，不承诺跨设备逐位一致。

### 一键执行顺序

1. 校验全部冻结数据和检查点哈希，保存本次配置快照。
2. 运行单步评测，保存原始预测、episode/工况指标及模型种子统计。
3. 对同一批起点运行两种递推模式和两种天气条件，覆盖 1–80 步；输出关键 H=1、5、10、20、40、80。
4. 按 episode 与固定训练种子汇总，自动保存逐 episode 配对差值与配对汇总。
5. 生成全部主实验表格和图形。
6. 核对 episode/起点身份、模型与 horizon 覆盖、数组轴、停止掩码，以及首步与单步评测的一致性。
7. 运行 24 项回归测试，保存测试日志。
8. 更新报告中带标记的“最近一次验收”段落，保留人工分析和独立扩展章节；保存结果资产哈希。

因此，一个配置复现 4 个模型族（Persistence、Linear、直接 MLP、残差 MLP）、8 个模型配置和 6 个关键 horizon，超过至少 3 个模型×4 个 horizon 的验收要求。它包含 18 个测试 episode，每回合 65 个共同起点，共 1170 个。原有 `--stage one_step`、`rollout`、`all` 保留，分别用于局部运行；`reproduce` 才执行完整报告流水线。

### 结果如何追溯

| 位置 | 如何使用 |
|---|---|
| results/one_step/predictions.csv.gz | 每条记录直接包含模型、episode_id、source_step、变量、真值、预测和输入 |
| results/rollout/origins.csv | 第 i 行映射到每个预测 NPZ 的起点轴 i，包含 episode_id 与原始 source_step |
| results/rollout/MODEL_predictions.npz | 起点×horizon×变量；每种模式/天气条件独立数组，另存 raw 与 attempted |
| results/rollout/validity_events.csv.gz | 从模型、episode_id、source_step 和 horizon 定位异常变量、幅度及停止事件 |
| results/rollout/metrics_by_episode.csv | 追踪聚合指标属于哪个回合和条件 |
| results/rollout/paired_episode_differences.csv.gz | 追踪同回合 A/B 模型差值及配对支持状态 |
| figures/ | 图、完整表格、绘图源数据及对齐检查记录 |

上表相对路径均相对于 `reports/week5/`。单条预测第 h 步对应原始转移 `source_step+h−1`，其目标状态为该转移的 next_state，避免把起点步与目标步混淆。

每次复现另建 `results/reproduction/运行时间编号/`，包含：

| 文件 | 内容 |
|---|---|
| run_manifest.json | 运行状态/时间、设备、Python/依赖版本、Git commit/dirty 状态、源代码哈希、输入与结果哈希及阶段耗时 |
| config_snapshot.yaml | 当次实际执行的配置原文 |
| code_snapshot/ | 当次评测、模型、指标、模拟器、数据代码、脚本和测试的源文件快照，覆盖未提交代码 |
| pipeline.log | 各阶段输出，便于定位失败 |
| tests.log | 本次回归测试输出 |
| verification.json | episode/起点数量、模型/horizon 覆盖及首步一致性验收 |
| report_snapshot.md | 本次完成时的报告副本 |

共享的 one_step、rollout、figures 目录保存最新结果，重跑会更新这些文件；运行配置、代码、报告快照和清单则按编号保留。清单中的共享结果哈希表示当次运行的文件内容，之后重跑可能覆盖；需要长期归档某轮完整原始预测时，应另外保存当轮结果目录。不会为每次重跑自动复制大体积数据集和全部预测文件。

Git dirty 状态会如实记录，不为完成本任务自动提交或推送。代码快照与哈希补足未提交改动的可追溯性。

### 验收边界

一键默认复现冻结的热工主实验。IB、投影模型和补充 episode-bootstrap 保留独立配置及命令，详见前面的扩展章节，不隐式混入主协议。概率校准未执行。回归测试通过与数值有限都不等于真实工业安全保证。

生成图形保留 PNG/PDF/SVG；面板对齐检查随绘图执行。此前的 PDF 字号/碰撞检查已完成，专用审计记录并不冒充每次主流水线都会重新运行的科学结论检查。
