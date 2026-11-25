"""固定初态和扰动的动作敏感性：真实模拟器与冻结模型逐计划配对，不训练模型。"""
from pathlib import Path
import sys
import json
import hashlib
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulators.simple_thermal import SimpleThermalEnv,ThermalConfig
from models.dynamics_base import load_dynamics
from evaluators.rollout import rollout_episode
from metrics.action_response import response_metrics


def simulate_plan(config,actions):
    """动作真实送入 env.step；每个计划从同一配置/种子重置，记录当前扰动与真实状态。"""
    env=SimpleThermalEnv(config); env.reset(seed=42)
    states=[env.state.copy()]; disturbances=[]
    for k,power in enumerate(actions):
        _,_,terminated,truncated,info=env.step([power])
        disturbances.append([info['outdoor_used_c'],info['internal_load_kw']])
        states.append(env.state.copy())
        if terminated or (truncated and k!=len(actions)-1): raise RuntimeError('Unexpected simulator end')
    env.close()
    return np.asarray(states),np.asarray(disturbances)


def main():
    cfg=yaml.safe_load((ROOT/'configs/experiments/week5_diagnostics.yaml').read_text())
    protocol=cfg['evaluation']; experiment=cfg['action_experiment']
    for relative,expected in protocol['sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected: raise ValueError('Frozen asset changed: '+relative)
    torch.set_num_threads(2)
    output=ROOT/cfg['output']/'action'; output.mkdir(parents=True,exist_ok=True)
    names=list(experiment['plans']); actions=np.stack([experiment['plans'][name] for name in names])
    if not np.isfinite(actions).all() or actions.min()<0 or actions.max()>5: raise ValueError('Illegal plan')
    models={spec['name']:load_dynamics(spec,ROOT,hashes=protocol['sha256']) for spec in protocol['models']}
    all_true=[]; all_d=[]; predictions={name:[] for name in models}; scenes=[]; metrics=[]; failures=[]
    for initial in experiment['initial_states']:
        for profile in experiment['weather_profiles']:
            scene=f'air{initial[0]:g}_wall{initial[1]:g}_{profile}'
            config=ThermalConfig(episode_steps=experiment['horizon'],initial_air_c=initial[0],initial_wall_c=initial[1],
                outdoor_mean_c=experiment['outdoor_mean_c'],outdoor_amplitude_c=0 if profile=='constant' else experiment['outdoor_amplitude_c'],
                internal_load_kw=experiment['load_kw'],process_noise_std=0,measurement_noise_std=0,weather_noise_std=0)
            true=[]; disturbances=None
            for plan in actions:
                states,d=simulate_plan(config,plan)
                if disturbances is not None: np.testing.assert_array_equal(disturbances,d)
                disturbances=d; true.append(states)
            true=np.stack(true); all_true.append(true); all_d.append(disturbances)
            scenes.append(dict(scene_id=scene,initial=initial,weather=profile,environment=asdict(config)))
            for spec in protocol['models']:
                name=spec['name']; model=models[name]; values=[]
                for index,plan in enumerate(actions):
                    result=rollout_episode(model,true[index,:-1],plan[:,None],disturbances,true[index,1:],
                        experiment['horizon'],'free_running','recorded_observation',
                        protocol['validity']['divergence_bounds_c'],protocol['validity'],asdict(config))
                    values.append(np.vstack([true[index,0],result['predicted'][0]]))
                    failures.append(dict(scene_id=scene,model=name,plan=names[index],failure_step=int(result['failure_step'][0]),reason=result['failure_reason'][0]))
                values=np.stack(values); predictions[name].append(values)
                family='residual_mlp' if name.startswith('residual_') else spec['kind']
                for plan,reference in experiment['references'].items():
                    i,j=names.index(plan),names.index(reference)
                    onset=int(np.flatnonzero(np.abs(actions[i]-actions[j])>1e-12)[0])
                    for v,variable in enumerate(('air','wall')):
                        response=true[i,:,v]-true[j,:,v]; predicted=values[i,:,v]-values[j,:,v]
                        np.testing.assert_allclose(predicted[:onset+1],0.,atol=1e-10,equal_nan=False)
                        row=response_metrics(response,predicted,onset,config.dt_hours*60,experiment['direction_epsilon_c'])
                        metrics.append(dict(scene_id=scene,weather=profile,initial_air_c=initial[0],initial_wall_c=initial[1],
                            model=name,family=family,seed=spec.get('seed',-1),plan=plan,reference=reference,
                            onset_action_step=onset,variable=variable,**row))
            print('Completed action scene '+scene,flush=True)
    saved=dict(true_state=np.stack(all_true),exogenous=np.stack(all_d),actions=actions)
    saved.update({name:np.stack(values) for name,values in predictions.items()})
    np.savez_compressed(output/'trajectories.npz',**saved)
    table=pd.DataFrame(metrics)
    table.to_csv(output/'response_metrics.csv',index=False)
    pd.DataFrame(failures).to_csv(output/'failures.csv',index=False)
    # 各场景先计算指标，再对场景等权；训练种子仍单独保留，未达到阈值不当作0。
    table['abs_peak_magnitude_error_c']=table.peak_magnitude_error_c.abs()
    table['abs_lag_error_minutes']=table.lag_error_minutes.abs()
    score_columns=['direction_accuracy','response_rmse_c','abs_peak_magnitude_error_c','abs_lag_error_minutes']
    agg={c+'_mean':(c,'mean') for c in score_columns}; agg.update(requested_scenes=('scene_id','size'),
        valid_response_scenes=('response_rmse_c','count'),valid_lag_scenes=('lag_error_minutes','count'))
    by_seed=table.groupby(['model','family','seed','plan','variable']).agg(**agg).reset_index()
    by_seed.to_csv(output/'response_by_seed.csv',index=False)
    agg={c+'_mean':(c+'_mean','mean') for c in score_columns}
    agg.update({c+'_seed_sd':(c+'_mean','std') for c in score_columns})
    agg.update(model_runs=('model','size'),valid_lag_scene_runs=('valid_lag_scenes','sum'),requested_scene_runs=('requested_scenes','sum'))
    by_seed.groupby(['family','plan','variable']).agg(**agg).reset_index().to_csv(output/'response_summary.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(config=cfg,scenes=scenes,plan_names=names,
        trajectory_axes=['scene','plan','state step 0..80','air/wall'],model_names=list(models),
        exogenous='same actual simulator outdoor/load sequence for every plan; no noise',
        scope='controlled simulator experiment, not a real-building experiment; independent extension'),indent=2))


if __name__=='__main__': main()
