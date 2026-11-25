"""动作响应实验：同初态、恒定扰动、无噪声环境与各动力学模型自由递推。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulators.simple_thermal import SimpleThermalEnv, ThermalConfig
from models.dynamics_base import PersistenceDynamics
from models.system_identification import LinearDynamics
from models.action_mlp import ActionMLP


def main():
    """环境真实状态和模型预测分别推进；模型只共享初始状态与动作/扰动序列。"""
    cfg = ThermalConfig(episode_steps=24, outdoor_amplitude_c=0, weather_noise_std=0,
        process_noise_std=0, measurement_noise_std=0,initial_air_c=20,initial_wall_c=19)
    output = ROOT/'reports/week4/results/action_response'
    output.mkdir(parents=True,exist_ok=True)
    models = {'Persistence':PersistenceDynamics(),
        'Linear':LinearDynamics.load(ROOT/'reports/week4/results/rollout/linear.npz')}
    training = json.loads((ROOT/'reports/week4/results/training/config.json').read_text())
    for run in training['runs']:
        models[f"MLP seed{run['seed']}"] = ActionMLP.load(ROOT/run['checkpoint'])
    scenarios = {'constant_0':np.zeros(24),'constant_2.5':np.full(24,2.5),
        'constant_5':np.full(24,5.),'step_0_to_5':np.r_[np.zeros(12),np.full(12,5.)]}
    rows = []
    fig, axes = plt.subplots(4,3,figsize=(16,14),constrained_layout=True)
    for row_index,(scenario,actions) in enumerate(scenarios.items()):
        env = SimpleThermalEnv(cfg)
        env.reset(seed=42)
        real = [env.state.copy()]
        predictions = {name:[env.state.copy()] for name in models}
        for step,action in enumerate(actions):
            env.step([action])
            real.append(env.state.copy())
            for name,model in models.items():
                predictions[name].append(model.predict_next(predictions[name][-1],[action],[cfg.outdoor_mean_c,cfg.internal_load_kw]))
        trajectories = {'Environment':np.array(real),**{name:np.array(values) for name,values in predictions.items()}}
        hours = np.arange(25)*cfg.dt_hours
        for name,states in trajectories.items():
            for step,state in enumerate(states):
                rows.append(dict(scenario=scenario,model=name,step=step,hour=hours[step],
                    air_c=state[0],wall_c=state[1],outdoor_c=cfg.outdoor_mean_c,
                    load_kw=cfg.internal_load_kw,action_kw=actions[step] if step<24 else None))
            for index in (0,1):
                axes[row_index,index].plot(hours,states[:,index],label=name,
                    color='black' if name=='Environment' else None,
                    linewidth=2.2 if name=='Environment' else 1.2,
                    linestyle='-' if name=='Environment' else '--')
        axes[row_index,2].stairs(actions,np.arange(25)*cfg.dt_hours,color='tab:red')
        axes[row_index,2].set(ylabel='Applied power (kW)',ylim=(-.2,5.3))
        for index,variable in enumerate(('Air','Wall')):
            axes[row_index,index].set(ylabel=f'{variable} temperature (°C)')
        for axis in axes[row_index]:
            axis.set(title=scenario,xlabel='Hours',xlim=(0,4))
            axis.grid(alpha=.2)
        env.close()
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Action response | initial air 20°C, wall 19°C | outdoor 6°C | load 0.3 kW | no noise')
    fig.savefig(output/'action_response.png',dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output/'trajectories.csv',index=False)
    (output/'config.json').write_text(json.dumps(dict(initial=[20,19],outdoor_c=6,
        load_kw=.3,steps=24,dt_hours=cfg.dt_hours,noise='disabled',seed=42,
        scenarios={name:actions.tolist() for name,actions in scenarios.items()},
        purpose='controlled action sensitivity experiment with fixed exogenous; four-hour recursive rollout'),indent=2))
    print(output)


if __name__ == '__main__':
    main()
