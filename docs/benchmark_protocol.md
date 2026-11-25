# Week 7 统一Benchmark配置契约 v1.0

## 任务1交付与入口

统一配置使用YAML，路径以项目根目录解析。`configs/benchmark/boptest_smoke.yaml`包含1个日期、Persistence/ARX两个检查点、8个有效控制步；`boptest_week6.yaml`包含11检查点、8测试日期和任务矩阵。清单为`configs/benchmark/manifests/boptest_week6.json`，记录18块训练/验证/测试身份、时间边界和逐文件SHA256。

```bash
conda activate mvp
python scripts/validate_benchmark.py --config configs/benchmark/boptest_smoke.yaml
python scripts/validate_benchmark.py --config configs/benchmark/boptest_week6.yaml
python -m unittest tests.test_benchmark_config
```

Week7 统一 Runner 已接入离线预检、模型加载、BOPTEST 回合、评测与独立产物目录。单独运行配置预检只检查本地资产，不执行仿真。Week6 原复现入口继续可用。

## 必填字段

| 字段 | 内容与约束 |
| --- | --- |
| schema_version | 字符串`1.0`；重复YAML键和未知顶层/主要节点字段报错 |
| model | instances逐项含name/kind/checkpoint.path/checkpoint.sha256；history_steps=16、device、观测/外生字段顺序与单位、输入12/输出6维 |
| env | BOPTEST URL、v0.9.0、完整commit、案例名、步长900秒、预热604800秒、历史上下文96步、有效评分步数、面积192m²、资产清单哈希和动作定义 |
| dataset | online/recorded、数据清单与哈希、训练/验证身份、测试日期、天气信息、上下文与固定动作来源 |
| seed | 每个模型训练种子；candidate基数604200加source_step；观测故障基数64042，开环加起点source_step，闭环加start_day |
| evaluator | 启用任务、teacher_forcing/free_running、起点步幅、任务条件矩阵、故障严重程度、embedded/none基线 |
| horizon | prediction_steps与mpc_steps分别定义；候选数、候选动作保持步数 |
| metrics | 指标、单位、成本权重、有效性阈值和episode汇总规则 |
| output | 相对项目输出目录、run名称、csv/npz/json/jsonl/md产物格式、新唯一run目录策略 |
| failure_policy | 非法预测停止并留前缀；规划失败释放覆盖；推进错误停止episode；只读重试次数与HTTP/规划超时 |

配置中的`model.instances`可以只有一个模型，也可以是一个比较矩阵。Persistence与ARX没有训练种子，seed.model中填null；神经模型42/43/44必须匹配其检查点元数据。checkpoint只读取本地文件并校验SHA256，核对kind与历史长度，不连接API。

## 模型输入、输出与单位

每个历史token按顺序含5个公开观测、1个实际/候选设定值、2个天气和4个日历量。观测顺序固定为操作温度、供水温度、回水温度、地板热流、总电功率；单位为degC/degC/degC/kW/kW。天气为室外degC与太阳辐照W/m2，日/年正余弦单位1。输出为下一5个公开观测和本15分钟区间电耗kWh。

API设定值单位K，模型和控制器单位degC；边界为实验18～24degC，元数据合法边界5～35degC。area用于将原生kWh/m²转换为kWh。任何将W、kW、kWh或K、degC混用的字段配置在离线预检报错。

`dataset.mode=recorded`表示开环采用已记录轨迹，闭环任务仍需真实BOPTEST服务。online表示短时在线采集目标。两者都通过manifest确定日期，并以`context_policy=replay_recorded_actions`重放同一96步上下文。开环未来动作来源`recorded_applied_actions`，允许固定动作计划已知。主天气信息为保持起点，ideal_api_reference可单独设为理想预报对照。未来室温不可作为外生信息。

## 时间、任务矩阵与失败语义

每个块先7天预热，再1天上下文，最后2天有效目标。预热、上下文和目标一起检查块间时间不重叠。smoke只评分其中8步，仍保留相同预热/上下文。预测步长必须在评分长度支持范围内；MPC预测时域独立配置。

任务矩阵逐项记录日期与条件。Week6模板的clean闭环使用8日期；鲁棒性闭环只使用160/292/340三个参考日期；开环故障有0.1/0.3degC和1/4步缺失，闭环仅运行0.3degC和4步缺失及goal_plus1。policy_ranking使用本次closed_loop实际动作前缀，因此须同时启用closed_loop。不同任务不会自动展开未配置的日期/条件组合。

供水传感器缺失按最后有效值填补。开环缺失在起点产生1/4步初始缺失，模型自由递推后不回填未来真实状态；闭环每24步产生指定连续缺失段。噪声污染三个公开温度通道。goal_plus1改变独立舒适边界1degC；模型权重冻结。

非法预测保存原值、首次事件及之前的前缀，后续标记无效；规划失败释放`oveTSet_activate`交还内置控制。HTTP推进可能已改变服务端状态，因此advance_requests必须为0次重试，发生失败后结束该episode。只读请求可以重试。planning_seconds只规定检查预算，不能保证抢占一个正在阻塞的模型调用。

## 指标与复现规则

温度误差以degC、功率误差以kW、区间能耗误差以kWh分别报告。成本为舒适区外温差平方×步长小时数×comfort权重，加本步电耗×energy权重，加设定值变化平方×action_change权重。舒适标准独立于可控动作。权重明确保存，不能把加权分数称为货币费用。

有效性边界包括zone0～50degC、水温0～90degC、候选步能耗0～10kWh、绝对发散阈值1000。开环物理异常保留原值，MPC可以据协议排除无效候选；全部无效时fallback。有效比例与误差同时报告。

先在完整episode内聚合窗口误差，再汇总固定训练种子与日期。训练种子、重叠窗口、15分钟时间步均不能充当独立物理episode。日期172保留为日历变化案例，冷季保留不冒称范围外天气，参考范围匹配也不保证联合ID。

输出产物包括 config.yaml、versions.json、input_hashes.json、metrics.json、trajectories.csv、predictions.npz、events.jsonl、run.log、status.json、failure.json（失败时）和 report.md。配置预检不创建run目录、不启动环境，校验失败CLI退出码2，错误包含字段路径。

## 验收范围

当前支持固定案例与既有12输入/6输出检查点契约。URL可配置，变更案例、单位、步长或模型维度须显式扩展契约与数据模型，不通过自动猜测转换。离线校验包括全部资产、数据文件与检查点哈希，数据身份/时间隔离、种子规则和动作范围。服务端版本/输入元数据的实时核对将在Runner连接后进行。


## 任务 2：统一模型加载

`benchmark/loaders.py` 的 `load_models(config, root)` 接受 YAML 路径或配置字典，返回按模型名称索引的模型字典。先复用完整配置预检，再复用 `models/dynamics_base.py` 的 `load_boptest_dynamics` 加载检查点；不启动模拟器。

```python
from benchmark.loaders import load_models
models = load_models("configs/benchmark/boptest_week6.yaml")
model = models["arx"]
# history 为物理单位的 (16, 12) 或 (N, 16, 12) 数组。
prediction = model.predict(history)
transition = model.predict_transition(history)
model.reset()
```

输出为 `(6,)` 或 `(N, 6)`：前五维是下一步公开观测，最后一维是本控制间隔能耗 kWh。`predict_transition` 分别返回 `next_observation` 和 `interval_energy_kwh`；能耗不能作为第六个状态递推。输入字段顺序、物理单位沿用本协议。

历史每次显式传入，`reset()` 不缓存历史；GRU 每个窗口从零隐藏状态开始。神经网络在配置的 CPU/CUDA 设备计算，输出统一转回 CPU NumPy；ARX 和 Persistence 使用 NumPy CPU 计算。请求不可用 CUDA 会报错，不静默回退。

加载阶段检查 SHA256、模型类型、历史长度、训练种子、12/6 维标准化向量、有限值与正缩放参数、ARX 系数 `(193, 6)`，并严格加载神经网络参数。最后用标准化均值构造单窗口及批量接口探测；这只是加载检查，不作为预测精度结果。输入错误或非有限输出保留明确异常。

验收命令：

```bash
python scripts/check_benchmark_models.py --config configs/benchmark/boptest_week6.yaml
python -m unittest tests.test_benchmark_loaders tests.test_benchmark_config
```

五类模型为 Persistence、ARX、历史 MLP、残差历史 MLP、GRU，共 11 个现有检查点。本任务复用已有权重，不重新训练模型。


## 任务 3：每次 run 的产物规范

`benchmark/artifacts.py` 提供 `RunArtifacts(config, root)`，先完成离线配置校验，再以 UTC 微秒时间、run 名称和随机 ID 创建独立目录。默认输出在 `reports/runs/`，同名配置重复运行也不会覆盖历史结果。它是产物写入组件，不会自行执行模型或模拟器；统一 Runner 已接入该组件。

| 文件 | 内容与用途 |
| --- | --- |
| config.yaml | 已解析并通过校验的完整配置快照 |
| versions.json | Python、系统、关键依赖版本，Git commit、工作区是否有修改及变更文件列表；不采集环境变量或代理凭据 |
| input_hashes.json | 配置、检查点、数据清单、逐 episode 数据及构建资产的实际 SHA256，初始化时再次核对 |
| metrics.json | 初始为 pending；写入后保留完整指标字典及 available 状态 |
| trajectories.csv | 逐行轨迹，首行冻结列集合，后续缺列或多列拒绝；失败前行保留 |
| predictions.npz | 可选原始预测数组、真实值、有效标记等，由调用者传入；保留 NaN/Inf，禁用 object 数组 |
| events.jsonl | 结构化事件、UTC 时间、级别和当时的窗口/回合上下文 |
| run.log | 可读事件日志；第三方 stdout/stderr 尚需后续 Runner 接管 |
| context.json | 最近一次 task/model/episode/origin/step 等上下文；由 Runner 更新 |
| status.json | running / completed / failed，开始结束时间、run ID 与已写轨迹行数 |
| failure.json | 失败时保存异常类型、消息、堆栈、最后上下文与已写行数 |
| artifacts.json | 终态产物的大小与 SHA256；排除自身及最后写入的 status，避免循环哈希 |
| report.md | 可选 Markdown 报告，由调用者提供 |

JSON/YAML 使用同目录临时文件写入并原子替换，防止读取半份状态；CSV/JSONL 每条写入即关闭文件。JSON 内非有限值编码为显式 `{nonfinite: nan/inf/-inf}` 标记，NPZ 和 CSV 保留原值，不丢弃失败样本。产物组件只记录调用者提供的数据，不承担指标计算或失败样本统计。

正常退出上下文时，至少有一条轨迹和非空指标才能标记 completed；completed 表示流程产物齐备，不表示所有预测有效。上下文内异常自动标记 failed，保存上下文后继续抛出异常，不伪装成成功。可通过 `set_context` 与 `log` 记录局部预测失败/fallback 后继续运行。强制杀进程、断电等无法执行异常处理的情况会留下 running 状态，不能当成已完成；自动恢复属于后续任务。

```python
from benchmark.config import load_config
from benchmark.artifacts import RunArtifacts

config = load_config('configs/benchmark/boptest_smoke.yaml')
with RunArtifacts(config) as run:
    run.set_context(task='closed_loop', model='arx', episode='id_test_day160', step=0)
    # 以下为调用示意，实际字段和值由后续 Runner 的评测结果提供。
    run.append_trajectory({'episode': 'id_test_day160', 'step': 0, 'zone_c': 21.0})
    run.write_metrics({'example_only': True, 'failure_count': 0})
```

验收：`python -m unittest tests.test_benchmark_artifacts`。测试目录自动清理，样例数值不作为 BOPTEST 实验结果。


## 任务 4：统一 Runner

先启动本地 BOPTEST 服务，再在项目根目录执行：

```bash
conda activate mvp
python run_benchmark.py --config configs/benchmark/boptest_smoke.yaml
```

终端打印本次独立目录；该目录包含配置、逐步轨迹、原始预测、指标、日志及报告。smoke 使用 Persistence、ARX 和内置控制器；修改配置中的模型列表及对应种子即可切换模型。完整矩阵可指定 `configs/benchmark/boptest_week6.yaml`，运行时间显著更长。

在线开环先在服务中重放已记录动作再预测；闭环模型每步规划，动作在服务中执行。历史上下文不计入评分；MPC 规划失败释放覆盖给内置控制器，推进失败不重试。`completed` 表示流程完成，应结合有效比例、fallback 和失败事件解读。预测误差按变量和 episode 保存，并汇总训练种子；排序比较冻结的实际动作计划。

时间预算只在模型预测调用之间检查；无法抢占正在执行的调用。服务版本和控制输入元数据在连接时检查，本地文件哈希在连接前检查。
