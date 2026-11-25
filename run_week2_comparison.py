# 总说明：运行三个配对场景，保存逐回合轨迹、统一指标、配置和温度/动作曲线。
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulators.simple_thermal import ThermalConfig
from controllers.mpc import MPCConfig
from evaluators.closed_loop import run_episode

def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def main():
    out=Path('reports/week2/results/paired_comparison'); out.mkdir(parents=True,exist_ok=True)
    scenarios=[{'name':'cold_seed42','seed':42,'initial_air_c':18.,'initial_wall_c':19.},
               {'name':'mild_seed43','seed':43,'initial_air_c':20.,'initial_wall_c':20.},
               {'name':'warm_seed44','seed':44,'initial_air_c':22.,'initial_wall_c':21.}]
    env_config=ThermalConfig(); mpc_config=MPCConfig()
    metadata={'scenarios':scenarios,'thermal_config':asdict(env_config),'mpc_config':asdict(mpc_config),
       'rule':{'target_c':env_config.target_c,'deadband_c':.5,'heating_power_kw':env_config.max_power_kw},
       'scoring':{'tracking_weight':1.0,'energy_weight':1.0,'definition':'sum((next_true_air-target)^2 + action_kw*dt_hours)'},
       'weather_information':'MPC holds current observed outdoor temperature constant over horizon',
       'timing':'wall time around each controller act call; measured separately for each controller',
       'model_note':'MPC uses true deterministic thermal equations with noisy observations and constant observed weather; environment adds process noise',
       'timeout_note':'checked between horizon steps, cannot preempt individual NumPy operation'}
    (out/'config.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summaries=[]; pair_deltas=[]
    for scenario in scenarios:
        pair={}; metrics={}
        for name in ('rule','mpc'):
            rows,summary,initial=run_episode(name,scenario,env_config,mpc_config)
            pair[name]=rows; metrics[name]=summary
            save_csv(out/f"{scenario['name']}_{name}_trajectory.csv",rows)
            summaries.append({'scenario':scenario['name'],'controller':name,'seed':scenario['seed'],
               'initial_air_c':scenario['initial_air_c'],'initial_wall_c':scenario['initial_wall_c'],
               **{k:v for k,v in summary.items() if k!='fallback_reasons'},'fallback_reasons':json.dumps(summary['fallback_reasons'],ensure_ascii=False)})
        # 严格核验配对场景的天气及两类噪声；控制器动作不同不应改变这些随机流。
        for key in ('outdoor_used_c','outdoor_next_c','process_air_c','process_wall_c','measurement_air_c','measurement_wall_c','measurement_outdoor_c'):
            np.testing.assert_array_equal([r[key] for r in pair['rule']],[r[key] for r in pair['mpc']])
        pair_deltas.append({'scenario':scenario['name'],**{f'{k}_mpc_minus_rule':metrics['mpc'][k]-metrics['rule'][k] for k in ('cumulative_cost','tracking_mae_c','energy_kwh','comfort_violation_rate')}})
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        for name in ('rule','mpc'):
            r=pair[name]; x=[v['hour'] for v in r]
            axes[0].plot(x,[v['true_air_after_c'] for v in r],label=name)
            axes[1].step(x,[v['action_kw'] for v in r],where='pre',label=name)
        axes[0].axhline(env_config.target_c,color='black',linestyle='--',label='target')
        axes[0].set(ylabel='True indoor (°C)',title=scenario['name']); axes[1].set(xlabel='Time (hours)',ylabel='Heating (kW)')
        for ax in axes: ax.legend(); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(out/f"{scenario['name']}_comparison.png",dpi=160); plt.close(fig)
    save_csv(out/'episode_metrics.csv',summaries); save_csv(out/'paired_differences.csv',pair_deltas)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
