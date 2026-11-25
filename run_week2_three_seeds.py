# 总说明：固定三个初始工况，各以 42、43、44 三个环境种子执行配对闭环并分别绘制种子平均。
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
    out=Path('reports/week2/results/scenario_three_seeds'); out.mkdir(parents=True,exist_ok=True)
    scenarios=[{'name':'cold','initial_air_c':18.,'initial_wall_c':19.},
               {'name':'mild','initial_air_c':20.,'initial_wall_c':20.},
               {'name':'warm','initial_air_c':22.,'initial_wall_c':21.}]
    seeds=[42,43,44]; env_cfg=ThermalConfig(); mpc_cfg=MPCConfig()
    config={'scenarios':scenarios,'environment_seeds':seeds,'mpc_planner_seed_fixed':mpc_cfg.seed,
            'thermal_config':asdict(env_cfg),'mpc_config':asdict(mpc_cfg),
            'rule':{'target_c':env_cfg.target_c,'deadband_c':0.5,'heating_power_kw':env_cfg.max_power_kw,'initial_heating_on':False},
            'metric':'sum((next true air - target)^2 + action * dt); both weights=1',
            'band':'mean +/- one population standard deviation (ddof=0), not confidence interval',
            'weather':'paired controllers receive same exogenous noise; MPC assumes current observed outdoor temperature persists'}
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summaries=[]; aggregated=[]; differences=[]
    for scenario in scenarios:
        runs={'rule':[],'mpc':[]}
        for seed in seeds:
            case={**scenario,'seed':seed}; pair={}
            for controller in ('rule','mpc'):
                rows,metrics,initial=run_episode(controller,case,env_cfg,mpc_cfg)
                pair[controller]=rows; runs[controller].append(rows)
                save_csv(out/f"{scenario['name']}_seed{seed}_{controller}.csv",rows)
                summaries.append({'scenario':scenario['name'],'seed':seed,'controller':controller,
                    **{k:v for k,v in metrics.items() if k!='fallback_reasons'},'fallback_reasons':json.dumps(metrics['fallback_reasons'],ensure_ascii=False)})
            # 同一场景、同一种子下，天气和过程/测量噪声严格一致。
            for key in ('outdoor_used_c','outdoor_next_c','process_air_c','process_wall_c','measurement_air_c','measurement_wall_c','measurement_outdoor_c'):
                np.testing.assert_array_equal([r[key] for r in pair['rule']],[r[key] for r in pair['mpc']])
        meanrows=[]; fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
        for controller,color in [('rule','tab:blue'),('mpc','tab:orange')]:
            times=np.array([r['hour'] for r in runs[controller][0]])
            for run in runs[controller][1:]: np.testing.assert_array_equal(times,[r['hour'] for r in run])
            arrays={key:np.array([[r[key] for r in run] for run in runs[controller]],dtype=float)
                    for key in ('true_air_after_c','action_kw')}
            air=arrays['true_air_after_c']; action=arrays['action_kw']
            # 阴影是三个独立环境种子之间的标准差，并非置信区间。
            for ax,arr,is_step in ((axes[0],air,False),(axes[1],action,True)):
                mean=arr.mean(axis=0); sd=arr.std(axis=0,ddof=0)
                ax.plot(times,mean,label=controller,color=color,drawstyle='steps-pre' if is_step else 'default')
                ax.fill_between(times,mean-sd,mean+sd,color=color,alpha=.16,step='pre' if is_step else None)
            for j,t in enumerate(times):
                meanrows.append({'hour':t,'controller':controller,'true_air_mean_c':air[:,j].mean(),'true_air_std_c':air[:,j].std(),
                                 'action_mean_kw':action[:,j].mean(),'action_std_kw':action[:,j].std()})
            selected=[r for r in summaries if r['scenario']==scenario['name'] and r['controller']==controller]
            metrics=('cumulative_cost','tracking_mae_c','energy_kwh','comfort_violation_rate','decision_mean_ms','fallback_count')
            aggregated.append({'scenario':scenario['name'],'controller':controller,'n_seeds':len(seeds),
                **{k+'_mean':float(np.mean([r[k] for r in selected])) for k in metrics},
                **{k+'_std':float(np.std([r[k] for r in selected],ddof=0)) for k in metrics}})
        for seed in seeds:
            pair={r['controller']:r for r in summaries if r['scenario']==scenario['name'] and r['seed']==seed}
            differences.append({'scenario':scenario['name'],'seed':seed,
                **{k+'_mpc_minus_rule':pair['mpc'][k]-pair['rule'][k] for k in ('cumulative_cost','tracking_mae_c','energy_kwh','comfort_violation_rate')}})
        axes[0].axhline(env_cfg.target_c,color='black',linestyle='--',label='Target')
        axes[0].set(ylabel='True indoor temperature (°C)',title=f"{scenario['name']}: initial air {scenario['initial_air_c']:g} °C, wall {scenario['initial_wall_c']:g} °C | seeds 42, 43, 44")
        axes[1].set(xlabel='Time (hours)',ylabel='Heating power (kW)')
        for ax in axes: ax.legend(); ax.grid(alpha=.25)
        fig.suptitle('Mean across three seeds; shading = ±1 population SD',fontsize=11)
        fig.tight_layout(); fig.savefig(out/f"{scenario['name']}_three_seed_mean.png",dpi=160); plt.close(fig)
        save_csv(out/f"{scenario['name']}_mean_trajectory.csv",meanrows)
    save_csv(out/'episode_metrics.csv',summaries); save_csv(out/'scenario_metrics.csv',aggregated); save_csv(out/'paired_differences.csv',differences)
    print(json.dumps(aggregated,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
