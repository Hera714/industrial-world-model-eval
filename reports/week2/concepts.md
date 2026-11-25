# Week 2：二阶热工环境

## 变量与单位

- 状态 $x_t = [T_{\mathrm{air},t}, T_{\mathrm{wall},t}]$：室内空气和墙体温度，单位 °C。
- 动作 $u_t = P_{\mathrm{heat},t}$：从时刻 $t$ 到 $t+1$ 持续施加的加热功率，单位 kW，范围 0～5 kW。
- 扰动 $d_t = [T_{\mathrm{out},t}, Q_{\mathrm{load}}]$：室外温度（°C）和固定内部热负荷（kW）。
- 观测 $o_t$：室温、墙温、室外温度分别加测量噪声；控制器不直接获得未加噪声的真实状态。
- 目标室温为 21 °C；舒适区为 20～23 °C，仅用于记录违规，不是安全硬约束。

## 状态方程

设置步长 $\Delta t=1/6$ 小时，热容量 $C_{\mathrm{air}}=2$、$C_{\mathrm{wall}}=12$ kWh/°C，热导 $G_{aw}=0.3$、$G_{ao}=0.15$、$G_{wo}=0.1$ kW/°C。每个动作在一个步长内保持不变：

$$
\begin{aligned}
T_{\mathrm{air},t+1} ={}& T_{\mathrm{air},t}
 + \frac{\Delta t}{C_{\mathrm{air}}}\left[
G_{aw}(T_{\mathrm{wall},t}-T_{\mathrm{air},t})
+G_{ao}(T_{\mathrm{out},t}-T_{\mathrm{air},t})
+P_{\mathrm{heat},t}+Q_{\mathrm{load}}\right]
+w_{\mathrm{air},t}, \\
T_{\mathrm{wall},t+1} ={}& T_{\mathrm{wall},t}
 + \frac{\Delta t}{C_{\mathrm{wall}}}\left[
G_{aw}(T_{\mathrm{air},t}-T_{\mathrm{wall},t})
+G_{wo}(T_{\mathrm{out},t}-T_{\mathrm{wall},t})\right]
+w_{\mathrm{wall},t}.
\end{aligned}
$$

室外温度按 24 小时正弦周期加随机扰动生成；数值为教学用设定，不是实际建筑参数。$w_{\mathrm{air},t}$ 和 $w_{\mathrm{wall},t}$ 是影响真实状态的过程噪声；测量噪声只影响观测。天气、过程和测量噪声使用由重置种子派生的独立随机流。

## 调用和边界

```python
from simulators.simple_thermal import SimpleThermalEnv
env = SimpleThermalEnv()
obs, info = env.reset(seed=42, options={'initial_air_c': 20, 'initial_wall_c': 19})
obs, reward, terminated, truncated, info = env.step([3.0])
```

观测顺序为 $[T_{\mathrm{air}}, T_{\mathrm{wall}}, T_{\mathrm{out}}]$；info 包含真实状态、动作、生效期间天气、过程噪声、测量噪声、能耗及舒适违规。示例 reward 为 -(温度跟踪误差平方 + 能耗数值)，这是教学权重，不是货币成本；不同单位的组合仅为人为评分。动作必须为合法范围内的单元素有限数组，非法动作抛 ValueError 且不推进状态。

本环境暂不建模安全终止：terminated 始终为 False；第 144 步因时长结束返回 truncated=True；结束后必须 reset。未定义真实建筑安全限值，不具备 fail-safe 控制。

测试：`python -m unittest discover -s tests -p test_env_contract.py`。Gymnasium 检查器对未归一化 kW 动作和无限观测空间发出建议警告；这是为保留物理单位、容纳高斯噪声而做的选择。
