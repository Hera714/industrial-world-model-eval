# 总说明：对比相同初始状态和随机扰动下，不同固定加热功率的 144 小时温度轨迹。
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulators.simple_thermal import SimpleThermalEnv, ThermalConfig

def main():
    # 10 分钟一步，864 步等于 144 小时；各组使用相同种子和初始条件。
    config=ThermalConfig(episode_steps=864)
    output=Path('reports/week2/results'); output.mkdir(parents=True,exist_ok=True)
    trajectories={}
    for power in (0.0,2.0,5.0):
        env=SimpleThermalEnv(config); obs,info=env.reset(seed=42)
        rows=[{'hour':0.0,'power_kw':power,'air_c':info['true_state_c'][0],'wall_c':info['true_state_c'][1],'outdoor_c':info['outdoor_c']}]
        for step in range(config.episode_steps):
            obs,reward,terminated,truncated,info=env.step([power])
            rows.append({'hour':(step+1)*config.dt_hours,'power_kw':power,'air_c':info['true_state_c'][0],'wall_c':info['true_state_c'][1],'outdoor_c':info['next_outdoor_c']})
            if terminated or truncated: break
        trajectories[power]=rows
    # 验证各组天气扰动一致，并保留逐步真实状态数据。
    baseline=np.array([r['outdoor_c'] for r in trajectories[0.0]])
    for rows in trajectories.values(): np.testing.assert_array_equal(baseline,[r['outdoor_c'] for r in rows])
    with (output/'thermal_heating_144h.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['hour','power_kw','air_c','wall_c','outdoor_c']); w.writeheader()
        for rows in trajectories.values(): w.writerows(rows)
    # 两幅子图分别显示室温、墙温，天气作为参考；不把曲线当作真实建筑结果。
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for power,rows in trajectories.items():
        t=[r['hour'] for r in rows]
        axes[0].plot(t,[r['air_c'] for r in rows],label=f'{power:g} kW')
        axes[1].plot(t,[r['wall_c'] for r in rows],label=f'{power:g} kW')
    axes[0].axhline(config.target_c,color='black',linestyle='--',label='Target 21 °C')
    axes[0].plot(t,baseline,color='gray',alpha=.45,linewidth=.8,label='Outdoor')
    axes[0].set(ylabel='Air temperature (°C)',title='Simulated heating response: identical weather and noise')
    axes[1].set(xlabel='Time (hours)',ylabel='Wall temperature (°C)')
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output/'thermal_heating_144h.png',dpi=160); plt.close(fig)
    for power,rows in trajectories.items(): print(f"{power:g} kW: 0h air={rows[0]['air_c']:.2f} °C, 144h air={rows[-1]['air_c']:.2f} °C, wall={rows[-1]['wall_c']:.2f} °C")
    print('Saved',output/'thermal_heating_144h.png')
if __name__=='__main__': main()
