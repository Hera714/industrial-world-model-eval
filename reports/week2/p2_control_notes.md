# Week 2 P2：强化学习、风险敏感 MPC 与部分可观测控制

> **边界说明：** 本文是阅读结论和后续实验设计；SAC、PPO、TD-MPC2 尚未在本仓库训练。当前已完成二阶热工环境、规则控制、random shooting MPC、线性辨识、LQR 与模拟安全机制。

## 0. 当前基线的位置

当前 random shooting MPC 从 $o_t=[T_{\mathrm{air}},T_{\mathrm{wall}},T_{\mathrm{out}}]$ 生成 $N=256$ 条、长度 $H=12$ 的合法加热功率序列；用已知无噪声热工方程或线性辨识模型预测；按温度跟踪、能耗、动作变化和舒适性软惩罚评分；只执行最优序列的第一个动作。

它是**有显式模型的在线规划器**：每一步模拟 $256\times12$ 次，默认未来室外温度保持当前观测值，没有读取真实未来天气。安全守卫会限幅或替换非法动作；舒适性在 MPC 内仍是软惩罚。

| 方法 | 本质 | 训练时需大量环境交互 | 部署时每步主要计算 | 项目定位 |
|---|---|---:|---|---|
| random shooting MPC（已实现） | 有模型、在线采样规划 | 否；辨识模型时需数据 | 候选轨迹仿真与打分 | 可解释基线 |
| SAC | 无模型、随机 actor-critic、off-policy | 是，但数据可复用 | 一次策略网络前向 | 连续控制 RL 对照 |
| PPO | 无模型、随机策略梯度、on-policy | 是；更新前需新轨迹 | 一次策略网络前向 | 透明的 RL 对照 |
| TD-MPC2 | 学习潜变量世界模型、价值和策略并规划 | 是 | 潜变量候选轨迹评估 | 世界模型扩展 |
| 鲁棒 MPC | 面向有界最坏情形的 MPC | 不一定 | 多情景/最坏情况优化 | MPC 安全扩展 |
| 机会约束 MPC | 面向违约概率上限的随机 MPC | 需误差样本或分布 | 概率约束或情景优化 | 先校准不确定性 |

---

## 1. SAC：随机策略、最大熵与 off-policy

SAC 同时学习随机策略 $\pi_\theta(a\mid o)$ 和动作价值函数 $Q_\phi(o,a)$。策略输出的是动作分布而非唯一加热功率；训练时从中采样，测试时可取均值或采样动作，再经过安全层。

最大熵目标为：

$$
J(\pi)=\mathbb E\left[\sum_{t=0}^{\infty}\gamma^t
\left(r_t+\alpha\mathcal H(\pi(\cdot\mid o_t))\right)\right].
$$

$r_t$ 是奖励，$\gamma$ 是折扣，$\mathcal H$ 是策略熵，$\alpha$ 权衡回报和探索。熵项避免训练中的策略过早坍缩到单一动作，不是把随机当作实际控制目标。SAC 原始定义就是最大熵、随机 actor-critic 与 off-policy 更新。[SAC 原论文](https://proceedings.mlr.press/v80/haarnoja18b.html)

每个交互转移 $(o_t,a_t,r_t,o_{t+1},\text{done})$ 都进入 replay buffer。之后可从较早策略收集的转移中抽样，反复更新当前网络；这就是 off-policy 数据复用，不表示可以不与环境交互。

本项目的奖励应严格对应当前代价：

$$
r_t=-\left[
w_e(T_{\mathrm{air},t}-T_{\mathrm{set},t})^2+
w_pP_{\mathrm{heat},t}\Delta t+
w_{\Delta u}(u_t-u_{t-1})^2+
w_v\operatorname{violation}_t
\right].
$$

| 维度 | 当前 shooting MPC | SAC |
|---|---|---|
| 决策来源 | 在线搜索一段动作序列 | 训练后 actor 直接映射 $o_t\mapsto\pi(a\mid o_t)$ |
| 动力学模型 | 需要已知或辨识模型 | 标准 SAC 不显式使用模型 |
| 数据使用 | 用模型规划 | replay buffer 反复抽样更新 |
| 未来扰动 | 显式规定预测边界 | 在训练分布中隐式学会应对 |
| 部署耗时 | 随候选数与时域增长 | 通常低，但训练前成本高 |
| 安全 | 需约束、守卫和 fallback | 同样需要外部安全机制 |

接入时应让训练覆盖多初始温度、天气与噪声种子，隔离测试种子；所有 actor 动作仍须经过安全守卫，并记录原始动作、守卫后动作和原因。

---

## 2. PPO：裁剪策略更新与 on-policy

PPO 先使用**当前策略**采一批新轨迹，再对该批数据做有限轮更新；更新后旧批次不能像 SAC replay buffer 一样长期使用。这是 on-policy 的关键。

新旧策略概率比和裁剪代理目标分别为：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)},
$$

$$
L^{\mathrm{CLIP}}(\theta)=
\mathbb E_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

其中 $\hat A_t$ 是优势估计。若新策略相对旧策略变化过大，裁剪会削弱这次更新带来的表面收益，降低一次梯度步把策略推得过远的风险。它不是物理安全约束，也不保证温度不越界。[PPO 原论文](https://arxiv.org/abs/1707.06347)

| 方面 | PPO | SAC |
|---|---|---|
| 数据属性 | on-policy：依赖刚收集的轨迹 | off-policy：可重用旧数据 |
| 主要稳定器 | 概率比裁剪 $\epsilon$ | 双 Q、目标网络、熵正则等 |
| 数据需求 | 每轮更新前继续仿真采样 | 旧轨迹利用率通常更高 |
| 部署 | 策略网络前向 + 安全层 | 策略网络前向 + 安全层 |

PPO 与 SAC 的部署通常比 MPC 快，但前提是已在足够广的情景分布中训练。分布外天气或模型变化发生时，不能默认策略能安全处理，必须保留限幅、状态监控和 rule-based fallback。

---

## 3. TD-MPC2：世界模型、价值估计和局部规划

TD-MPC2 是模型式 RL：在潜空间学习可供控制使用的隐式世界模型，并在此做局部轨迹优化。论文将其描述为在学习到的无解码器世界模型潜空间中进行局部轨迹优化。[TD-MPC2 原论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/cf73d57b6dcda32b293df7c2d5341f49-Paper-Conference.pdf)

可用下列教学化符号理解：

$$
z_t=e_\theta(o_t),\qquad
z_{t+1}=f_\theta(z_t,a_t),\qquad
\hat r_t=R_\theta(z_t,a_t).
$$

同时学习策略先验 $\pi_\theta(a\mid z)$ 与价值估计 $V_\theta(z)$ 或 $Q_\theta(z,a)$。候选序列评分含短期预测奖励与末端价值：

$$
\hat G_t=\sum_{k=0}^{H-1}\gamma^k\hat r_{t+k}
+\gamma^H\hat V(z_{t+H}).
$$

末端价值使短时域规划不会把时域后的影响直接当成零；当前 12 步 shooting MPC 没有该学习到的长期价值项。

### 3.1 与当前 MPC 的数据流

~~~mermaid
flowchart LR
  subgraph A[当前 Random Shooting MPC：已实现]
    AO[带噪观测 o_t] --> AC[直接取 air、wall、outdoor]
    AC --> AS[均匀采样 256 条 H=12 动作序列]
    AS --> AD[已知热工式 或 线性辨识模型]
    AD --> AJ[跟踪+能耗+变动+软舒适成本]
    AJ --> AA[最优序列首动作]
    AA --> AG[安全守卫 / fallback]
    AG --> AE[真实热工环境]
    AE --> AO
  end
  subgraph B[TD-MPC2：未来扩展，尚未实现]
    BO[观测 o_t] --> BE[编码器 e：z_t]
    BE --> BP[策略先验引导候选序列]
    BP --> BD[学习到的潜动力学 f]
    BD --> BR[预测奖励 R + 末端价值 V/Q]
    BR --> BA[首动作]
    BA --> BG[安全守卫 / fallback]
    BG --> BENV[真实环境]
    BENV --> BO
    BENV -. 交互转移 .-> BT[replay buffer]
    BT -. 联合训练 e、f、R、V/Q、策略 .-> BE
  end
~~~

TD-MPC2 不是有预测模型就必然优于普通 MPC：模型、价值和策略都须从交互数据学习，训练分布外仍可能失效。对这个低维、物理关系已知的热工环境，已知模型更可解释；TD-MPC2 更适合在高维观测、未知模型或多任务时检验。公平比较须对齐动作范围、种子、扰动可见性、真实环境成本和安全层，不能只给 TD-MPC2 真实未来天气。

---

## 4. 鲁棒 MPC：把最坏情况写进优化

现有 MPC 是名义 MPC：一个模型、一个未来天气假设。鲁棒 MPC 显式定义模型误差和扰动集合：

$$
x_{k+1}=f_{\mathrm{nom}}(x_k,u_k,d_k)+w_k,
\qquad w_k\in\mathcal W,
$$

并对最坏允许情形优化：

$$
\min_{u_{t:t+H-1}}\max_{w_{t:t+H-1}\in\mathcal W}
J(x,u,w)
\quad
\text{s.t.}\quad
x_k\in\mathcal X,\;u_k\in\mathcal U
\quad\forall w\in\mathcal W.
$$

保证只相对于声明的集合 $\mathcal W$ 有效；集合越大，控制越保守，也可能没有可行候选。

**本项目最小版本：**

1. 用系统辨识验证集残差确定误差界，或定义三条天气情景 $T_{\mathrm{out}}^{\mathrm{nom}}+\{-\delta,0,+\delta\}$。
2. 每条候选动作序列均在三个情景 rollout，评分取最大成本，并淘汰任一情景触及硬安全边界的候选。
3. 和名义 MPC 在同一初始状态、天气、过程噪声、目标和种子下配对评测。
4. 同时报平均/最坏成本、能耗、舒适违规率、安全终止、fallback 与规划耗时。能耗上升但极端天气违规减少，是保守性的正常代价。

只采若干随机噪声轨迹并观察到没有出事，并不是鲁棒保证。模型、扰动不确定性以及跨时域概率约束都应在问题定义中明确。[Li 等，2000](https://www.sciencedirect.com/science/article/pii/S0098135400003987)

---

## 5. Chance-constrained MPC：用概率量化违约风险

确定性硬约束要求每次预测满足 $20\le T_{\mathrm{air},k}\le23$。机会约束允许有限风险：

$$
\Pr\left(T_{\mathrm{air},k}\in[20,23]\right)
\ge1-\epsilon_k.
$$

若关心整个时域都不违约，应使用联合机会约束：

$$
\Pr\left(T_{\mathrm{air},t+1:t+H}\in[20,23]^H\right)
\ge1-\epsilon.
$$

逐步各自有 $90\%$ 满足概率，并不推出 12 步同时有 $90\%$ 满足概率。实现可做风险分配，使 $\sum_k\epsilon_k\leq\epsilon$；也可用情景采样，或在高斯近似且校准可信时做约束收紧。

| 方案 | 不确定性假设 | 约束含义 | 风险 |
|---|---|---|---|
| 名义 MPC | 单一预测 | 名义轨迹满足 | 忽略误差 |
| 鲁棒 MPC | 误差在有界集合内 | 所有允许误差均满足 | 可能过保守 |
| 机会约束 MPC | 误差分布或样本已校准 | 满足概率至少 $1-\epsilon$ | 分布失配会使实际违规超标 |

正确口径是：先只在验证集统计一步/多步预测误差分位数，预先固定 $\epsilon$ 和约束收紧；再在隔离测试种子上报告实际覆盖率、舒适违规率、能耗与收紧宽度。不能看到测试违规后调大区间又仍把它称为测试结果。概率约束也不等价于真实设备安全认证。

---

## 6. 部分可观测控制：隐藏墙温的可执行实验

目前控制器可见 $o_t=[T_{\mathrm{air}},T_{\mathrm{wall}},T_{\mathrm{out}}]$，墙温带测量噪声但仍可见。若墙体没有传感器，真实状态仍为

$$
x_t=[T_{\mathrm{air},t},T_{\mathrm{wall},t}],
$$

公开观测则只有

$$
o_t=[T_{\mathrm{air},t}+v_{\mathrm{air},t},\;
T_{\mathrm{out},t}+v_{\mathrm{out},t}].
$$

控制器必须依据历史 $h_t=(o_0,a_0,\ldots,o_t)$ 估计墙温；这就是 POMDP，不能偷读真实 $T_{\mathrm{wall}}$。

~~~mermaid
flowchart LR
  X[真实状态 x = air, wall] -->|测量噪声；仅公开 air 和 outdoor| O[观测 o]
  X -->|真实动力学 + 过程噪声| XN[下一真实状态]
  O --> EST[状态估计器：历史、动作、模型]
  U[上一动作] --> EST
  EST --> Z[估计 x_hat = air_hat, wall_hat]
  Z --> C[MPC / LQR / RL policy]
  C --> G[安全守卫]
  G -->|动作 u| XN
  X -. 仅评测日志可见，控制器不可读 .-> EVAL[真实误差、约束、估计误差]
~~~

| 层级 | 控制器拿到的墙温 | 用途 | 可否作为真实部署结果 |
|---|---|---|---|
| Oracle 全状态 MPC/LQR | 真实或直接测得的墙温 | 性能上界、消融 | 墙温无传感器时不可以 |
| 朴素部分观测 | 令 $\hat T_{\mathrm{wall}}=T_{\mathrm{air}}$ | 最低基线 | 可以 |
| 估计器 + 部分观测 | 从 $o_{0:t},u_{0:t-1}$ 估计 | 正式可部署对照 | 可以 |

**最小实现步骤：**

1. 为环境增加配置 **observe_wall=False**，使观测固定为 $[T_{\mathrm{air}},T_{\mathrm{out}}]$；真实墙温只写入评测日志，不能传给控制器。
2. 采用已知线性化模型：从 $(\hat x_t,u_t,T_{\mathrm{out},t})$ 预测一步，再按室温观测残差校正空气和墙温估计。第一版可用 Kalman filter；若只用固定增益，应诚实称为 observer。
3. 将 $[\hat T_{\mathrm{air}},\hat T_{\mathrm{wall}},T_{\mathrm{out}}]$ 送入现有 MPC/LQR，安全守卫不变。
4. 按初始温度 $18/20/22\ ^\circ\mathrm C$、种子 $42/43/44$ 配对评测，保证真实天气、过程噪声、目标和动作边界一致。
5. 记录真实状态、可见观测、估计状态、创新项、估计误差、闭环成本、舒适违规、安全事件和耗时；绝不以真实墙温事后修正控制输入。

$$
\operatorname{RMSE}_{\mathrm{wall}}=
\sqrt{\frac1T\sum_{t=1}^{T}
(\hat T_{\mathrm{wall},t}-T_{\mathrm{wall},t})^2}.
$$

真正要检验的是墙温估计更准是否带来更低闭环成本或更少违规，而不是先假定它必然成立。

---

## 7. P2 交付与推荐顺序

本文件已交付六个主题的概念、公式、边界、当前 MPC 对照、TD-MPC2 数据流图和隐藏墙温实验规范。推荐顺序：

1. 部分可观测估计器，先验证代码没有混用观测和真实状态。
2. 鲁棒/机会约束 shooting MPC，复用已有候选采样和日志，得到风险—能耗曲线。
3. SAC，在安全包装后的环境上建立端到端 RL 基线，保持测试场景未参与训练。
4. PPO，使用同一奖励、训练预算和隔离测试集，比较 on-policy 数据需求。
5. TD-MPC2，待日志规范、信息边界和安全评测固定后再引入。

无论采用哪种学习控制器，模拟器结果都不能直接外推为真实楼宇或工业系统的安全保证；当前安全机制只是实验保护层。
