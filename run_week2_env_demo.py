"""Run a reproducible fixed-action episode and save all state/noise fields."""
# 总说明：运行固定加热功率的可复现示例，保存完整状态、观测和噪声轨迹。
import csv
from pathlib import Path
from simulators.simple_thermal import SimpleThermalEnv

def main():
    out=Path('reports/week2/results'); out.mkdir(parents=True,exist_ok=True)
    # 固定种子 42，保证天气和噪声可复现。
    env=SimpleThermalEnv(); obs,initial=env.reset(seed=42); rows=[]
    # 每步施加 2 kW，并逐项收集真实状态、观测、噪声和成本。
    for _ in range(env.config.episode_steps):
        obs,reward,terminated,truncated,info=env.step([2.0])
        rows.append({'step':info['step'],'air_before_c':info['state_before_c'][0],'wall_before_c':info['state_before_c'][1],
        'air_c':info['true_state_c'][0],'wall_c':info['true_state_c'][1],'outdoor_used_c':info['outdoor_used_c'],
        'action_kw':info['action_kw'],'internal_load_kw':info['internal_load_kw'],'observed_air_c':obs[0],
        'observed_wall_c':obs[1],'observed_outdoor_c':obs[2],'process_air_c':info['process_noise_c'][0],
        'process_wall_c':info['process_noise_c'][1],'measurement_air_c':info['measurement_noise_c'][0],
        'measurement_wall_c':info['measurement_noise_c'][1],'measurement_outdoor_c':info['measurement_noise_c'][2],
        'target_c':info['target_c'],'energy_kwh':info['energy_kwh'],'cost':info['cost'],'comfort_violation':info['comfort_violation'],
        'terminated':terminated,'truncated':truncated})
        if terminated or truncated: break
    # 将逐步记录保存为 CSV，供检查和绘图。
    with (out/'fixed_action_seed42.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print('Saved',len(rows),'steps to',out/'fixed_action_seed42.csv')
if __name__=='__main__': main()
