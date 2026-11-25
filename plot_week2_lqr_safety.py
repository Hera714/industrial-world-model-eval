# 总说明：从实际 LQR/MPC 回合生成分场景统计、动作限幅图和中文实验报告。
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=Path('reports/week2/results/lqr_safety')
    cfg=json.loads((out/'config.json').read_text()); df=pd.read_csv(out/'episode_metrics.csv')
    lines=['# Week 2：LQR 与模拟安全后备','',
    '## 工作点与离散 LQR','',
    '固定室外温度 6 °C，室温目标 21 °C，计算墙温稳态 17.25 °C、稳态加热功率 3.075 kW；功率位于 0～5 kW 范围。模型为线性仿射方程，围绕该点写成偏差系统。','',
    '$$\nQ=\\begin{bmatrix}1&0\\\\0&0.1\\end{bmatrix},\\qquad R=0.1,\\qquad K=\\begin{bmatrix}2.39897&0.36802\\end{bmatrix}.\n$$','',
    '使用离散代数 Riccati 方程得到反馈增益，原始动作按 0～5 kW 限幅。限幅后的反馈不再是严格无约束 LQR 最优解；只用带噪状态观测，固定工作点天气，没有未来天气前馈。完整矩阵 A、B、Q、R、K 见配置文件。','',
    '## 配对闭环：三个种子','',
    '每个初始温度对应三个固定环境种子 42、43、44；同一种子下控制器的天气和噪声相同。表中是三次回合的均值 ± 总体标准差；成本为相同的环境示例评分，不是电费。','',
    '| 初始温度 | 控制器 | 累计评分 | 跟踪 MAE °C | 能耗 kWh | 舒适违规率 |','|---|---|---:|---:|---:|---:|']
    for air in (18,20,22):
      for kind in ('lqr','mpc'):
        g=df[(df.initial_air_c==air)&(df.controller==kind)]
        f=lambda k,scale=1:f"{g[k].mean()*scale:.3f} ± {g[k].std(ddof=0)*scale:.3f}"
        lines.append(f"| {air} °C | {kind} | {f('cumulative_cost')} | {f('tracking_mae_c')} | {f('energy_kwh')} | {f('comfort_violation_rate',100)}% |")
      fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
      for kind in ('lqr','mpc'):
        runs=[pd.read_csv(out/f'air{air}_seed{seed}_{kind}.csv') for seed in (42,43,44)]
        t=runs[0].hour.to_numpy()
        for ax,key in ((axes[0],'true_air_after_c'),(axes[1],'action_kw')):
          x=np.stack([r[key].to_numpy() for r in runs]); mean=x.mean(axis=0); sd=x.std(axis=0)
          ax.plot(t,mean,label=kind,drawstyle='steps-pre' if key=='action_kw' else 'default')
          ax.fill_between(t,mean-sd,mean+sd,alpha=.12,step='pre' if key=='action_kw' else None)
      axes[0].axhline(21,color='black',linestyle='--',label='target'); axes[0].set(ylabel='Indoor temperature (°C)',title=f'Initial {air} °C: three-seed mean ± population SD')
      axes[1].set(xlabel='Time (hours)',ylabel='Heating power (kW)')
      for ax in axes: ax.legend(); ax.grid(alpha=.2)
      fig.tight_layout(); fig.savefig(out/f'air{air}_comparison.png',dpi=150); plt.close(fig)
    for air in (18,20,22):
      lines.extend(['',f'![初始室温 {air} °C 的三种子曲线](results/lqr_safety/air{air}_comparison.png)'])
    lqr=df[df.controller=='lqr']
    lines+=['',f"LQR 九个回合共出现 {int(lqr.lqr_clipped_count.sum())} 次原始动作限幅；单步原始和实际动作保存在每个轨迹 CSV。所有正常回合均完成 144 步，故障注入与正常回合分开。",'',
    '## 安全与故障注入','',
    '舒适区 20～23 °C 与教学安全边界 12～30 °C 分开。控制前依据带噪观测：低于安全下限请求 5 kW、高于上限请求 0 kW；非法观测或动作请求 0 kW；有限越界动作限幅。环境推进后若真实室温越界，包装层标记自然终止并记录事件。初始越界在第一步检查，不能保证避免单步跨越或真实安全。环境本体仍对非法动作报错。','',
    '故障注入日志记录了高温、低温、非法动作、限幅、无有效候选、超时和模型异常。MPC 时间预算仅在预测步之间检查；异常记录类型和消息，并返回规则后备。安全边界是演示参数，不可用于真实设备。','',
    '见 [实验配置](results/lqr_safety/config.json)、[故障注入](results/lqr_safety/failure_injection.json)、[逐回合指标](results/lqr_safety/episode_metrics.csv) 与同目录原始轨迹。','',
    '## 复现','', '在项目目录执行：','',
    '    conda activate mvp','    python run_week2_lqr_safety.py','    python plot_week2_lqr_safety.py','    python -m unittest discover -s tests','']
    Path('reports/week2/lqr_safety.md').write_text('\n'.join(lines),encoding='utf-8')
    print(df.groupby('controller')[['cumulative_cost','tracking_mae_c','energy_kwh','lqr_clipped_count','safety_interventions']].mean().to_string())
if __name__=='__main__': main()
