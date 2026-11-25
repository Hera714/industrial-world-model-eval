# Week 3 主环境：Siemens Industrial Benchmark

选择 IB Python 版，沿用 Week 2 的连续控制评测流程，并引入部分可观测、动作延迟和随机动态。BOPTEST 留作第二平台。

## 版本与复现

选定上游发布包 `industrial-benchmark-python==2.0`。官方仓库提供同版本 wheel。上游安装元数据声明 Gym、NumPy 和 SciPy，README 列出 Python 3.7，安装元数据允许 Python ≥3.6。项目目标是 WSL2 / Python 3.11，兼容性及实际依赖版本留待官方样例验证。

```bash
python -m pip install 'industrial-benchmark-python==2.0'
python -m pip show industrial-benchmark-python gym numpy scipy
```

根目录 LICENCE.txt 标注 Apache-2.0，Python wrapper 源码头及包元数据标注 MIT。保留原始许可和版权声明。发布包 SHA256 与对应源码提交待获取后补录。

## 变量与时间

| 变量 | 含义 | 范围 | 控制器可见 | 可控 |
|---|---|---|---|---|
| setpoint | 工况设定 | 0–100 | 是 | 仅初始化配置 |
| velocity | 速度操纵量 | 0–100 | 是 | 通过增量 |
| gain | 增益操纵量 | 0–100 | 是 | 通过增量 |
| shift | 偏移操纵量 | 0–100 | 是 | 通过增量 |
| fatigue | 疲劳指标 | wrapper 声明 0–1000 | 是 | 间接受影响 |
| consumption | 消耗指标 | wrapper 声明 0–1000 | 是 | 间接受影响 |
| cost | 上游成本，classic 奖励等于其相反数 | 以源码输出为准 | 通过奖励 | 间接受影响 |
| 潜变量与历史内部状态 | 模拟器内部动态 | 依变量而定 | 不进策略观测 | 无直接控制 |

上表前六项是 classic 观测顺序。使用基准数值尺度，上游未指定物理单位。连续动作按 velocity、gain、shift 顺序输入三个归一化增量命令，每项在 [-1,1]。一次决策推进一个仿真步，动作重复数为 1。回合上限暂定 1000 步，沿用 wrapper 默认值。没有物理采样周期，时间轴记录 step。

来源：[官方仓库及许可](https://github.com/siemens/industrialbenchmark)、[包元数据](https://github.com/siemens/industrialbenchmark/blob/master/setup.py)、[IBGym 源码](https://github.com/siemens/industrialbenchmark/blob/master/industrial_benchmark_python/IBGym.py)。
