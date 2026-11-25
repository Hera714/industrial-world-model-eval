# 总说明：在热工模拟环境中运行 24 小时规则控制，保存逐步数据、配置和控制曲线。
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulators.simple_thermal import SimpleThermalEnv
from controllers.rule_based import HysteresisController

def main():
    out=Path('reports/week2/results/rule_based'); out.mkdir(parents=True,exist_ok=True)
    env=SimpleThermalEnv(); c=env.config
    controller=HysteresisController(target_c=c.target_c,deadband_c=.5,heating_power_kw=c.max_power_kw)
    obs,initial=env.reset(seed=42); controller.reset()
    rows=[]; cumulative=0.0
    # 动作只由当前带测量噪声的观测决定，再实际送入模拟环境。
    for step in range(c.episode_steps):
        observed_before=obs.copy(); action=controller.act(observed_before)
        if not env.action_space.contains(action): raise ValueError('控制器输出动作不合法')
        obs,reward,terminated,truncated,info=env.step(action); cumulative+=info['cost']
        rows.append({'step':step+1,'hour':(step+1)*c.dt_hours,
          'air_before_c':info['state_before_c'][0],'wall_before_c':info['state_before_c'][1],
          'observed_air_before_c':observed_before[0],'observed_wall_before_c':observed_before[1],
          'observed_outdoor_before_c':observed_before[2], 'true_air_after_c':info['true_state_c'][0],
          'true_wall_after_c':info['true_state_c'][1], 'observed_air_after_c':obs[0],
          'observed_wall_after_c':obs[1], 'observed_outdoor_after_c':obs[2],
          'outdoor_used_c':info['outdoor_used_c'],'outdoor_next_c':info['next_outdoor_c'],
          'internal_load_kw':info['internal_load_kw'],'target_c':info['target_c'],'action_kw':info['action_kw'],
          'process_air_c':info['process_noise_c'][0],'process_wall_c':info['process_noise_c'][1],
          'measurement_air_after_c':info['measurement_noise_c'][0],
          'measurement_wall_after_c':info['measurement_noise_c'][1],
          'measurement_outdoor_after_c':info['measurement_noise_c'][2],
          'energy_kwh':info['energy_kwh'],'tracking_cost':info['tracking_cost'],'cost':info['cost'],
          'cumulative_cost':cumulative,'comfort_violation':info['comfort_violation'],
          'terminated':terminated,'truncated':truncated})
        if terminated or truncated: break
    with (out/'trajectory.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary={'seed':42,'steps':len(rows),'duration_hours':len(rows)*c.dt_hours,'initial_true_state_c':initial['true_state_c'].tolist(),
             'target_c':c.target_c,'lower_threshold_c':c.target_c-.5,'upper_threshold_c':c.target_c+.5,
             'initial_heating_on':False,'heating_power_kw':c.max_power_kw,
             'energy_kwh':sum(r['energy_kwh'] for r in rows),'cumulative_cost':cumulative,
             'tracking_mae_c':float(np.mean([abs(r['true_air_after_c']-c.target_c) for r in rows])),
             'comfort_violation_rate':float(np.mean([r['comfort_violation'] for r in rows])),
             'terminated':rows[-1]['terminated'],'truncated':rows[-1]['truncated'],
             'cost_definition':'(next true air - target)^2 + power * dt; illustrative unit-weighted score'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    # 绘图使用动作生效结束时刻；包含温度、目标、动作及逐步/累计代价。
    hours=[r['hour'] for r in rows]; fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    axes[0].plot(hours,[r['true_air_after_c'] for r in rows],label='True indoor'); axes[0].plot(hours,[r['observed_air_after_c'] for r in rows],alpha=.45,label='Observed indoor')
    axes[0].axhline(c.target_c,linestyle='--',color='black',label='Target'); axes[0].axhline(c.target_c-.5,linestyle=':',color='gray',label='Switch thresholds'); axes[0].axhline(c.target_c+.5,linestyle=':',color='gray')
    axes[0].set(ylabel='Temperature (°C)',title='Hysteresis control: 24-hour episode'); axes[0].legend()
    axes[1].step(hours,[r['action_kw'] for r in rows],where='pre'); axes[1].set(ylabel='Heating power (kW)',ylim=(-.3,c.max_power_kw+.3))
    axes[2].plot(hours,[r['cost'] for r in rows],label='Step cost'); axes[2].plot(hours,[r['cumulative_cost'] for r in rows],label='Cumulative cost'); axes[2].set(xlabel='Time (hours)',ylabel='Illustrative cost'); axes[2].legend()
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'control_trajectory.png',dpi=160); plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
