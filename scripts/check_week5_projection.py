"""复算投影模型每步未修正输出，保留修正前后轨迹和触发计数；原模型结果不覆盖。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.dynamics_base import load_dynamics

cfg=yaml.safe_load((ROOT/'configs/experiments/week5_projection.yaml').read_text())
folder=ROOT/cfg['output']/'rollout'
origins=pd.read_csv(folder/'origins.csv')
frame=pd.read_csv(ROOT/cfg['data']['directory']/'test.csv',skipinitialspace=True)
frame.columns=frame.columns.str.strip(); frame['episode_id']=frame.episode_id.str.strip()
states=[]; actions=[]; disturbances=[]
for origin in origins.itertuples():
    ep=frame[frame.episode_id.eq(origin.episode_id)].sort_values('source_step')
    section=ep.iloc[origin.source_step:origin.source_step+cfg['max_horizon']]
    states.append(np.stack(section.state.map(json.loads)))
    actions.append(np.stack(section.action_applied.map(json.loads)))
    disturbances.append(np.stack(section.exogenous_available_at_decision.map(json.loads)))
states,actions,disturbances=map(np.stack,(states,actions,disturbances))
counts=[]
for spec in cfg['models']:
    if spec['kind']!='rate_projection': continue
    model=load_dynamics(spec,ROOT,hashes=cfg['sha256']); saved={}
    with np.load(folder/f'{spec["name"]}_predictions.npz') as predictions:
        for condition in cfg['information']['conditions']:
            for mode in cfg['modes']:
                key=condition+'__'+mode
                corrected=predictions[key]
                previous=states if mode=='teacher_forcing' else np.concatenate([states[:,:1],corrected[:,:-1]],axis=1)
                d=disturbances.copy()
                if condition=='persistence_forecast': d[:,:,0]=d[:,:1,0]
                shape=corrected.shape
                recomputed,detail=model.predict_with_diagnostics(previous.reshape(-1,2),actions.reshape(-1,1),d.reshape(-1,2))
                np.testing.assert_allclose(recomputed.reshape(shape),corrected,atol=1e-5)
                raw=detail['raw'].reshape(shape); triggered=detail['triggered'].reshape(shape)
                saved[key+'__before']=raw; saved[key+'__after']=corrected; saved[key+'__triggered']=triggered
                counts.append(dict(model=spec['name'],mode=mode,condition=condition,
                    requested_origin_steps=shape[0]*shape[1],corrected_steps=int(triggered.any(axis=-1).sum()),
                    corrected_components=int(triggered.sum()),max_correction_c=float(np.abs(raw-recomputed.reshape(shape)).max()),
                    max_replay_difference_c=float(np.abs(recomputed.reshape(shape)-corrected).max())))
    np.savez_compressed(folder/f'{spec["name"]}_projection_diagnostics.npz',**saved)
pd.DataFrame(counts).to_csv(folder/'projection_summary.csv',index=False)
print(pd.DataFrame(counts).to_string(index=False))
