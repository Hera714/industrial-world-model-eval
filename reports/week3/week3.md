# Week 3 工业模拟器基线实验

使用官方样例同类的均匀随机策略，每步独立采样三个 [-1,1] 增量动作。工况为 50，每回合 1000 步，环境与策略种子均为 42、43、44。

| 种子 | 步数 | 原生累计奖励 | 累计成本 | 平均成本 | 平均疲劳 | 平均消耗 | 结束原因 |
|---|---|---|---|---|---|---|---|
| 42 | 1000 | -223630.837 | 223630.837 | 223.631 | 31.023 | 130.562 | time_limit |
| 43 | 1000 | -253701.932 | 253701.932 | 253.702 | 28.810 | 167.273 | time_limit |
| 44 | 1000 | -229991.473 | 229991.473 | 229.991 | 49.568 | 81.288 | time_limit |

## 三回合汇总

均值 ± 样本标准差（ddof=1），实验单位为完整回合。

- native_return 为 -235774.747 ± 15847.788
- cumulative_cost 为 235774.747 ± 15847.788
- mean_cost 为 235.775 ± 15.848
- mean_fatigue 为 36.467 ± 11.400
- mean_consumption 为 126.374 ± 43.145

## 指标口径与复现

原生逐步成本为 3 × fatigue + consumption，classic 奖励为其相反数。累计成本、平均成本、平均疲劳和平均消耗由逐步输出计算，保留上游基准尺度。原生累计奖励等于累计成本的相反数。

官方示例统计每个 100 步片段的折扣奖励（折扣 0.97），并除以 100。本报告统计完整 1000 步无折扣累计奖励，两者统计区间和折扣定义不同。

运行命令为 `python scripts/run_week3.py`。配置、软件版本和上游提交见 results/industrial_benchmark/baseline/config.json；逐回合 CSV、汇总 JSON 和三份 JSONL 轨迹位于同目录。

所有回合按时间上限截断。日志保留动作前观测、动作后公开状态、原生奖励、成本、诊断字段和空间检查结果。

## BOPTEST KPI 详解与核对表

以下口径依据 BOPTEST v0.9.0 源码（提交 `9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9`）。当前配置从模拟秒 86400 开始评测，到 172800 结束，共 96 个 900 秒控制步。预热期为 0–86400 秒，其作用是建立合理初始状态。累计 KPI 从评测起点计算到查询时已经到达的仿真时间，预热不计入。

| 原始字段 | API 单位 | 计算及归一化 | 本次统计窗口 | API 实测值 |
|---|---|---|---|---|
| `ener_tot` | kWh/m² | 案例指定 HVAC 能源输入功率积分后合计，再除以楼面面积 | 全天累计 | 0.2555181493 |
| `cost_tot` | $/m² 或 €/m² | 按能源载体将功率和对应时刻单价相乘后积分，再除以楼面面积 | 全天累计 | 0.0647738509 |
| `tdis_tot` | K·h/zone | 低于下限与高于上限的温差积分，合计后除以参与计算的区域数 | 全天累计 | 0.5439467736 |
| `pele_tot` | kW/m² | 同一时刻电功率求和，按 15 分钟分箱取样本均值，再取最大值并除以面积 | 全天峰值 | 0.0170552950 |
| `pgas_tot` | kW/m² | 燃气功率按 15 分钟分箱取均值后取峰值，再按面积归一化 | 全天峰值 | null |
| `pdih_tot` | kW/m² | 区域供热功率按 15 分钟分箱取均值后取峰值，再按面积归一化 | 全天峰值 | null |
| `emis_tot` | kgCO₂e/m² | 功率乘排放因子并积分，合计后除以面积 | 全天累计 | 0.0426715309 |
| `idis_tot` | ppm·h/zone | 超过空气质量上限的浓度偏差积分，按参与计算的区域数归一化 | 全天累计 | 0 |
| `time_rat` | s/s | 实际步间等待时间与模拟控制步长之比取平均 | 评测期内可用间隔 | 4.514552 × 10⁻⁶ |

本地 BOPTEST v0.9.0 的内置控制基线已跑完 96 步，期末查询一次 KPI。上表取自原始响应，显示值有舍入，完整精度见 [summary.json](results/boptest/baseline/summary.json)，逐步记录见 [episode_42.jsonl](results/boptest/baseline/episode_42.jsonl)。`null` 保留原样，与数值 0 分开。API 数值没有附币种标签，费用按原始货币单位列示。

案例配置面积为 192 m²。由官方面积指标反推，全天 HVAC 总能耗约为 **49.06 kWh**（0.2555181493 × 192），电力峰值约为 **3.27 kW**（0.0170552950 × 192）。这是单位反归一化。独立复算需另取完整的功率、价格、舒适边界与匹配的时间采样。

复现时从项目根目录运行 `no_proxy=127.0.0.1,localhost python scripts/run_week3.py --config configs/environments/boptest.yaml`。

### 四个常用指标怎么算

- **能耗**。若某设备以 2 kW 连续运行 30 分钟，它消耗 1 kWh。若案例用于归一化的楼面面积是 100 m²，对应 0.01 kWh/m²。用官方 `ener_tot` 反推总 kWh 时，应乘官方采用的面积。热泵的供热输出与热泵的电输入是不同测点，复算需采用案例 KPI 指定的能源输入。
- **能源成本**。若上述 1 kWh 的电价是 0.20 €/kWh，该段费用为 0.20 €，面积归一化后为 0.002 €/m²。电价随时间变化时，要先逐时刻将功率与价格对齐，再积分。原生能源费用也应与控制器自定义的奖励分别读取。
- **热舒适偏差**。假设舒适下限是 21 °C，房间在 19 °C 保持 30 分钟，则产生 1 K·h 的偏差。超上限也按超出的温差累计。温差 1 K 等于温差 1 °C。这个指标同时计入偏差幅度和持续时间。
- **电力峰值**。先把同一时刻的 HVAC 电功率相加，再按 15 分钟分箱。假设最高一箱的均值是 3 kW，面积 100 m²，峰值为 0.03 kW/m²。即使控制步也是 15 分钟，一个控制步结束时读到的瞬时功率也只能代表该采样时刻。

### 本地复算与验收

回合结束后原样保存 `GET /kpi/{testid}` 返回字段和数值。若要独立核对，先用 `GET /measurements/{testid}` 核实单位，并从案例配置与 KPI 测点映射确认面积、区域和能源测点。再通过 `PUT /results/{testid}` 获取评测窗口时序，确认拿到的采样和边界覆盖与服务端一致。能耗与费用采用源码的梯形积分，功率为 W、时间为秒时，除以 3,600,000 换成 kWh。舒适度需对齐随时间变化的上下限，温差按秒积分后除以 3600 得 K·h。峰值需复现相同的 15 分钟分箱与均值处理。

当前基线以原生 KPI 为验收值。逐步轨迹只有控制步末采样，独立复算待取得与服务端匹配的时序、价格和上下限后进行。「四个常用指标怎么算」中的数字是教学示例。

[官方 API](https://ibpsa.github.io/project1-boptest/docs-userguide/api.html) · [v0.9.0 KPI 源码](https://github.com/ibpsa/project1-boptest/blob/v0.9.0/kpis/kpi_calculator.py)

## 技术附录 · 从 Modelica 到 FMU，再到 Python 仿真

**Modelica** 描述物理规律，例如室温如何随供热和室外温度改变。导出工具会展开组件连接、生成求解所需的代码，并将模型打包成 **FMU**。FMU 是一个可解包的归档文件，通常包含 `modelDescription.xml`、适用平台的二进制和可选资源，例如天气文件。XML 描述变量名、输入输出、单位、FMI 版本及模型支持的接口。

**FMI** 规定宿主如何与模型交互。Model Exchange 模式由宿主提供积分器，并处理仿真步进与事件。Co-Simulation 模式下，FMU 自己处理通信点之间的内部推进，宿主负责定时交换输入输出。比如每隔 900 秒给 FMU 一次控制输入，FMU 内部仍可用更小的求解步。BOPTEST 把这层仿真封装在服务里，评测脚本通过 HTTP 发送控制并读取测量。

**FMPy** 是 Python 端的 FMU 加载与仿真工具。可用 `read_model_description` 先看 FMI 版本、接口类型和变量。做开环冒烟测试时，用 `simulate_fmu` 一次跑完。做 FMI 2.0 Co-Simulation 闭环时，通常依次实例化、初始化、写输入、调用 `doStep`、读输出，循环推进后再终止和释放。每次决策使用当前时刻可见的输出，下一时刻输出用于记录结果。

### 可选独立验证

从 FMPy 教程下载 Linux 版 `Rectifier.fmu`，将它放在当前目录。在安装了 FMPy 的 Python 环境中执行：

```python
from fmpy import read_model_description, simulate_fmu

filename = "Rectifier.fmu"
description = read_model_description(filename)
print(description.modelName, description.fmiVersion)
result = simulate_fmu(filename)
print(result.dtype.names)
print(result[[0, -1]])
```

验收时检查模型名称、FMI 版本、输出列，以及首末时间是否向前推进。WSL 需要 Linux 版 FMU。这个示例可独立验证 Python 加载和推进 FMU 的能力。目前未运行该示例。

[FMI 官方标准](https://fmi-standard.org/) · [FMPy 官方教程与示例下载](https://fmpy.readthedocs.io/en/latest/tutorial/)
## 整个流程
从你的项目出发，可以把整条流程分成两段：**上游先把楼宇做成可运行的模型，你再通过 BOPTEST 控制这个模型做实验。** 你现在主要在做第二段。

```text
上游制作楼宇模型                          你的 Week 3 实验

Modelica 源模型
    ↓ 用建模工具编译、导出
FMU 文件
    ↓ BOPTEST 服务按 FMI 接口加载
可推进的楼宇仿真 ←── HTTP API ──→ run_week3.py
    ↓                               ↓
更新室温、设备功率等                 保存轨迹、获取 KPI
```

### 1. 制作案例时，用 Modelica，并打包一次

案例作者用 **Modelica** 描述建筑和设备，例如墙体如何传热、热泵如何供暖、房间温度如何变化。模型写好后，建模工具把它编译并导出成一个 **`.fmu` 文件**。这一步就是打包。

对你选的案例，上游已经做好了这一步。仓库里有现成的 `wrapped.fmu`。因此你现在执行 `docker compose up` 时，主要是在构建和启动**运行这个模型的服务**。以后只换控制策略，也可以继续用同一个 FMU。只有你要改模型本身，例如更换热泵模型或修改墙体结构，才通常需要回到建模工具并重新导出。

### 2. 服务运行模型时，用 FMI

**FMU 是文件，FMI 是操作该文件里模型的接口标准。** BOPTEST 服务加载 FMU 后，通过 FMI 初始化模型、设置输入、推进模拟时间、读取输出。你写的 `run_week3.py` 通过 HTTP 请求服务，FMI 调用发生在服务内部。

以一次 15 分钟的控制步为例：

| 你的 Python 脚本做什么 | BOPTEST 服务内部做什么 |
|---|---|
| 调用 `POST /advance`。内置基线发送空对象，保留模型自带控制 | 推进 FMU 900 秒。这期间模型计算墙体传热、室温变化和设备运行 |
| 收到 JSON，记录步末室温等测点 | 读取测点，保留用于计算 KPI 的时序 |
| 重复 96 步后，调用 `GET /kpi` | 返回评测期的能耗、费用、热舒适偏差和电力峰值等指标 |

第一步 `advance` 之前，脚本还会先**选案例、设置控制步长、初始化并预热**。以后换成自己的控制器，变化主要发生在 `advance` 前：先根据当前观测算出动作，再把允许覆盖的控制量随请求发进去。

### 3. FMPy 是另一条运行路径

如果目的只是理解 **FMU 怎么被加载和推进**，可以让 Python 用 **FMPy 直接加载一个 FMU**，按 FMI 读写变量并推进仿真。这条路径不经过 BOPTEST 的 HTTP 服务，也不会自动得到 BOPTEST 的那套 KPI。

所以你当前要做的顺序是：**启动 Docker 服务 → 选用上游现成的楼宇 FMU → 通过 API 跑完 96 步 → 导出轨迹和 KPI**。FMPy 小实验可以独立做，用来亲手验证 BOPTEST 在服务内部替你完成的那一层。

### 实测 · BOPTEST 案例 FMU 独立运行

在 WSL Linux x86-64 下用 FMPy 0.3.32 加载 `bestest_hydronic_heat_pump/models/wrapped.fmu`。模型声明 FMI 2.0 Co-Simulation，包含 `linux64/wrapped.so`，室温输出为 `reaTZon_y`（K）。严格元数据验证拒绝了部分内部变量的 local/constant 组合，本次仅对该固定案例使用 `validate=False`。从 Ubuntu 官方 libgfortran4 包提取 `libgfortran.so.4` 到临时目录，并通过 `LD_LIBRARY_PATH` 提供给本次进程。

| 模拟秒 | 室温 K | 室温 °C |
|---:|---:|---:|
| 86400 | 293.1500 | 20.0000 |
| 87300 | 292.4327 | 19.2827 |
| 88200 | 292.2593 | 19.1093 |
| 89100 | 292.2506 | 19.1006 |
| 90000 | 292.2998 | 19.1498 |

模型从 86400 秒直接初始化，以 900 秒间隔输出，共验证四个推进区间。脚本为 `scripts/check_boptest_fmu.py`。FMPy 试验从指定时间直接初始化。上方的 HTTP 基线是另一次实验，由 BOPTEST 服务执行一天预热及 96 步评测后生成原生 KPI，两次试验的初始条件与用途不同。

## P2 · 仿真平台选型与现场接口对照

| 平台 | 控制入口 | 部署复杂度 | 时间步 | 复现实验要固定的配置 |
|---|---|---|---|---|
| EnergyPlus | Python Runtime API 注册回调，Data Exchange API 读测点、写 actuator | 较高，需安装引擎、准备建筑与天气文件、确定回调和测点 | 区域步由模型配置决定，HVAC 系统可能采用更短的内部步；在指定回调处决策 | 引擎版本、IDF/epJSON、EPW、运行期、区域步、回调位置、测点与执行器映射 |
| Sinergym | Gymnasium 的 `reset/step`，由封装对接 EnergyPlus 测点与动作 | 中等，需匹配 Sinergym 和 EnergyPlus，已有环境提供模型及控制封装 | 以环境配置的仿真步和动作执行规则为准 | 两个软件版本、环境 ID、建筑/天气、运行日期、步长、观测/动作、奖励、扰动与种子 |
| CityLearn | `CityLearnEnv.reset/step`，按配置控制单楼或多楼设备、储能 | 预置案例较低，主要使用 Python 与数据集；自建街区需配置设备和输入数据 | schema 的 `seconds_per_time_step` 与数据索引共同决定，常用数据为小时尺度 | 包版本、schema 和数据版本、模拟区间和步长、建筑选择、设备参数、集中/分布式控制、奖励与种子 |

EnergyPlus 适合调整建筑物理模型和控制挂点；Sinergym 适合沿用 Gymnasium 评测流程做 EnergyPlus 控制实验；CityLearn 适合街区协调与储能调度。三个平台的步数不能直接等同，跨平台结果需要同时记录实际时长、决策频率和 KPI 口径。

[EnergyPlus Python API](https://energyplus.readthedocs.io/en/latest/api.html) · [Sinergym 官方仓库](https://github.com/ugr-sail/sinergym) · [CityLearn 官方仓库](https://github.com/intelligent-environments-lab/CityLearn)

| 协议 | 如何取得测点值 | 单位和质量状态 | 怎样区分只读与可写 |
|---|---|---|---|
| OPC UA | 用 NodeId 定位变量，主动 Read 或订阅变化 | 工程量可由 `EngineeringUnits` 提供；记录 `DataValue.StatusCode`（Good、Uncertain、Bad）及可用的源时间戳 | 查看 `AccessLevel` 与当前用户的 `UserAccessLevel`，根据实际写入状态与读回确认结果 |
| BACnet | 按设备实例、对象类型和实例号定位，通过 ReadProperty 读取 `Present_Value`，支持时订阅 COV | 模拟量对象可有 `Units`；同时读取 `Status_Flags`，支持时读 `Reliability` | 依据对象属性和设备点表确认写权限；可命令点需记录写入优先级，并读回当前生效值 |
| Modbus | 按设备地址、功能码和寄存器地址轮询，按设备说明解码 | 单位、倍率、有符号性与多寄存器字序来自厂商寄存器表；通信异常、超时与设备诊断需单独记录，普通寄存器没有通用质量字段 | 输入寄存器与离散输入只读；保持寄存器或线圈是否允许写，还要核对厂商点表和设备配置 |

接入日志至少保留点位 ID、原始值、工程值、单位、时间戳及时间来源、质量状态和可写权限。下发控制时另记请求值、写入结果和读回生效值。传感器观测与控制命令分别入表，避免把写入成功误记成设备已达到目标。

[OPC UA 地址空间规范](https://reference.opcfoundation.org/specs/OPC-10000-3/5) · [BACnet 标准补充文件](https://bacnet.org/wp-content/uploads/sites/4/2022/08/Add-135-2010aa.pdf) · [Modbus 官方规范](https://www.modbus.org/modbus-specifications)

### Week 3 验收记录

- **IB**：官方样例在固定上游提交和记录过的 NumPy 兼容映射下已运行。标准脚本用 42、43、44 各跑完 1000 步，保留了逐步轨迹、配置和指标。动作边界、非法动作、时间截断和结束后重置已经过测试。
- **BOPTEST**：切换环境配置后复用 `run_standard_episode` 的采集循环。适配器处理各平台的接口差异，BOPTEST 的测点和原生 KPI 由专用导出逻辑保存。本地 v0.9.0 内置控制回合已跑满 96 步。P2 表格中的其他平台和现场协议为选型资料。
