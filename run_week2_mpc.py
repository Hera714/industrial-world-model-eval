# 总说明：在独立运行的热工环境中执行 MPC，记录 24 小时轨迹、规划耗时及后备原因。
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulators.simple_thermal import SimpleThermalEnv
from controllers.mpc import RandomShootingMPC

def main():
    out=Path('reports/week2/results/mpc'); out.mkdir(parents=True,exist_ok=True)
    env=SimpleThermalEnv(); controller=RandomShootingMPC(thermal=env.config)
    obs,initial=env.reset(seed=42); controller.reset(); rows=[]; cumulative=0.0
    # 控制器仅收到观测；环境以自己的随机扰动推进，真实状态仅用于记录和评测。
    for step in range(env.config.episode_steps):
        observed_before=obs.copy(); action,decision=controller.act(observed_before)
        if not env.action_space.contains(action): raise ValueError('MPC 输出越界')
        obs,reward,terminated,truncated,info=env.step(action); cumulative+=info['cost']
        rows.append({'step':step+1,'hour':(step+1)*env.config.dt_hours,'air_before_c':info['state_before_c'][0],
          'wall_before_c':info['state_before_c'][1],'observed_air_before_c':observed_before[0],
          'observed_wall_before_c':observed_before[1],'observed_outdoor_before_c':observed_before[2],
          'true_air_after_c':info['true_state_c'][0],'true_wall_after_c':info['true_state_c'][1],
          'observed_air_after_c':obs[0],'observed_wall_after_c':obs[1],'observed_outdoor_after_c':obs[2],
          'outdoor_used_c':info['outdoor_used_c'],'outdoor_next_c':info['next_outdoor_c'],'internal_load_kw':info['internal_load_kw'],
          'target_c':info['target_c'],'action_kw':info['action_kw'],'process_air_c':info['process_noise_c'][0],
          'process_wall_c':info['process_noise_c'][1],'measurement_air_after_c':info['measurement_noise_c'][0],
          'measurement_wall_after_c':info['measurement_noise_c'][1],'measurement_outdoor_after_c':info['measurement_noise_c'][2],
          'energy_kwh':info['energy_kwh'],'tracking_cost':info['tracking_cost'],'cost':info['cost'],
          'cumulative_cost':cumulative,'comfort_violation':info['comfort_violation'],
          'planning_time_s':decision['planning_time_s'],'fallback_reason':decision['fallback_reason'] or '',
          'terminated':terminated,'truncated':truncated})
        if terminated or truncated: break
    with (out/'trajectory.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    durations=np.array([r['planning_time_s'] for r in rows])
    summary={'seed_env':42,'initial_state_c':initial['true_state_c'].tolist(),'steps':len(rows),
      'duration_hours':len(rows)*env.config.dt_hours,'cumulative_environment_cost':cumulative,
      'tracking_mae_c':float(np.mean([abs(r['true_air_after_c']-r['target_c']) for r in rows])),
      'energy_kwh':sum(r['energy_kwh'] for r in rows),'comfort_violation_rate':float(np.mean([r['comfort_violation'] for r in rows])),
      'planning_time_mean_ms':float(durations.mean()*1000),'planning_time_p95_ms':float(np.quantile(durations,.95)*1000),
      'fallback_count':sum(bool(r['fallback_reason']) for r in rows),
      'fallback_reasons':dict(Counter(r['fallback_reason'] for r in rows if r['fallback_reason'])),
      'terminated':rows[-1]['terminated'],'truncated':rows[-1]['truncated'],
      'weather_assumption':'hold current observed outdoor temperature fixed across horizon',
      'cost_note':'environment score is tracking squared plus energy; MPC objective additionally includes action-change and comfort penalties',
      'timeout_note':'checked between vectorized horizon steps; cannot interrupt a running NumPy operation',
      'thermal_config':asdict(env.config),'mpc_config':asdict(controller.config)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    # 三张子图分别展示真实/观测温度、实际执行动作和环境评分。
    hours=[r['hour'] for r in rows]; fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    axes[0].plot(hours,[r['true_air_after_c'] for r in rows],label='True indoor'); axes[0].plot(hours,[r['observed_air_after_c'] for r in rows],alpha=.5,label='Observed indoor')
    axes[0].axhline(env.config.target_c,color='black',linestyle='--',label='Target'); axes[0].set(ylabel='Temperature (°C)',title='Random shooting MPC: 24-hour closed loop'); axes[0].legend()
    axes[1].step(hours,[r['action_kw'] for r in rows],where='pre'); axes[1].set(ylabel='Power (kW)',ylim=(-.3,env.config.max_power_kw+.3))
    axes[2].plot(hours,[r['cost'] for r in rows],label='Step environment score'); axes[2].plot(hours,[r['cumulative_cost'] for r in rows],label='Cumulative environment score'); axes[2].set(xlabel='Time (hours)',ylabel='Illustrative score'); axes[2].legend()
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'control_trajectory.png',dpi=160); plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
