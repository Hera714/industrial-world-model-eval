"""用相同观测信息和随机场景比较已知、线性与 MLP 动力学 MPC。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.system_identification import LinearDynamics
from models.action_mlp import ActionMLP
from evaluators.closed_loop import run_episode
from controllers.mpc import MPCConfig
from simulators.simple_thermal import ThermalConfig


class ThermalPlanningModel:
    """将 MPC 的当前室外温度扩展为模型需要的两维扰动向量。"""
    def __init__(self, model, load):
        self.model, self.load = model, load

    def predict_next(self, state, action, outdoor):
        exogenous = np.broadcast_to([outdoor,self.load],(len(state),2))
        return self.model.predict_next(state,action,exogenous)


def main():
    """三种初态各配对三个环境种子，MLP 在 CPU 上做小批量规划。"""
    torch.set_num_threads(2)
    cfg = ThermalConfig()
    mpc = MPCConfig()
    training = json.loads((ROOT/'reports/week4/results/training/config.json').read_text())
    models = {'known':None, 'linear':LinearDynamics.load(ROOT/'reports/week4/results/rollout/linear.npz')}
    for run in training['runs']:
        models[f"mlp_seed{run['seed']}"] = ActionMLP.load(ROOT/run['checkpoint'])
    output = ROOT/'reports/week4/results/closed_loop'
    residual_config = ROOT/'reports/week4/results/training_residual/config.json'
    if residual_config.exists():
        for run in json.loads(residual_config.read_text())['runs']:
            models[f"residual_seed{run['seed']}"] = ActionMLP.load(ROOT/run['checkpoint'])
    output.mkdir(parents=True,exist_ok=True)
    summaries, trajectories = [], []
    for air,wall in ((18,19),(20,20),(22,21)):
        for seed in (42,43,44):
            scenario = dict(name=f'air{air}_seed{seed}',seed=seed,initial_air_c=air,initial_wall_c=wall)
            reference = None
            for name,model in models.items():
                dynamics = None if model is None else ThermalPlanningModel(model,cfg.internal_load_kw)
                rows,summary,_ = run_episode('mpc',scenario,cfg,mpc,dynamics)
                frame = pd.DataFrame(rows)
                noise_columns = ['outdoor_used_c','process_air_c','process_wall_c','measurement_air_c','measurement_wall_c']
                if reference is not None:
                    np.testing.assert_array_equal(reference,frame[noise_columns].to_numpy())
                reference = frame[noise_columns].to_numpy()
                frame.to_csv(output/f"{scenario['name']}_{name}.csv",index=False)
                frame['model'],frame['initial_air_c'],frame['seed'] = name,air,seed
                trajectories.append(frame)
                summaries.append(dict(model=name,initial_air_c=air,seed=seed,
                    **{key:value for key,value in summary.items() if key!='fallback_reasons'},
                    fallback_reasons=json.dumps(summary['fallback_reasons'])))
    metrics = pd.DataFrame(summaries)
    metrics.to_csv(output/'episode_metrics.csv',index=False)
    columns = ['cumulative_cost','tracking_mae_c','energy_kwh','comfort_violation_rate','decision_mean_ms','decision_p95_ms','fallback_count']
    grouped = metrics.groupby(['initial_air_c','model'])[columns].agg(['mean','std'])
    grouped.to_csv(output/'summary.csv')
    # 每个初态分别绘制，曲线为三个环境种子均值。
    all_rows = pd.concat(trajectories)
    fig,axes = plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for index,air in enumerate((18,20,22)):
        for name in models:
            frame = all_rows[(all_rows.initial_air_c==air)&(all_rows.model==name)]
            curve = frame.groupby('hour')[['true_air_after_c','action_kw']].mean()
            axes[index,0].plot(curve.index,curve.true_air_after_c,label=name)
            axes[index,1].plot(curve.index,curve.action_kw,label=name,alpha=.8)
        axes[index,0].axhline(21,color='black',linestyle='--',linewidth=.8)
        for axis in axes[index]:
            axis.set(xlabel='Hours',title=f'Initial air {air}°C')
            axis.grid(alpha=.2)
        axes[index,0].set_ylabel('Air temperature (°C)')
        axes[index,1].set_ylabel('Power (kW)')
    axes[0,0].legend(fontsize=8)
    fig.savefig(output/'control_comparison.png',dpi=150)
    plt.close(fig)
    print(metrics.groupby('model')[['cumulative_cost','tracking_mae_c','energy_kwh','fallback_count']].mean().to_string())


if __name__ == '__main__':
    main()
