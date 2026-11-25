"""IB 独立历史评测：20步窗口、零初始化GRU、递推公开观测；复用两层统计。"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.dynamics_base import load_dynamics
from evaluators.rollout import rollout_errors, summarize_rollout_metrics


def ib_window(observations, actions, starts, step, mode, recursive_window):
    """自由递推只读调用方维护的窗口；teacher forcing 才能读取当前真实历史。"""
    if mode == 'free_running':
        return recursive_window
    if mode != 'teacher_forcing':
        raise ValueError('Unknown mode')
    return np.stack([np.concatenate([observations[t+step-19:t+step+1],actions[t+step-19:t+step+1]],axis=-1) for t in starts])


def predict_ib(model, window):
    """每个窗口独立计算；非有限原输出与没有返回输出的推理异常分别处理。"""
    if model is None:
        return window[:, -1, :6].copy(), ['']*len(window)
    try:
        value = np.asarray(model.predict_window(window))
        if value.shape != (len(window),6):
            raise ValueError('Invalid shape')
        return value, ['']*len(window)
    except Exception:
        value = np.full((len(window),6), np.nan); errors = ['']*len(window)
        for i in range(len(window)):
            try:
                result = np.asarray(model.predict_window(window[i]))
                if result.shape != (6,):
                    raise ValueError('Invalid shape')
                value[i] = result
            except Exception as exc:
                raw = getattr(exc,'prediction',None)
                if raw is not None and np.asarray(raw).shape == (6,):
                    value[i] = raw
                else:
                    errors[i] = f'{type(exc).__name__}: {exc}'
        return value, errors


def save_ib_validity(logs, events, output):
    """保留所有请求轨迹，并将声明空间越界与真正停止失败分别汇总。"""
    frame=pd.DataFrame(logs)
    keys=['model','mode','episode_id','source_step']
    frame['failed']=frame.failure_step>0
    frame['first_anomaly_step']=0
    for event in ('nan','inf','inference_error','observation_bounds','divergence'):
        frame[event+'_steps']=0
    if events:
        event_frame=pd.DataFrame(events)
        first=event_frame.groupby(keys).horizon.min().rename('first_anomaly_step')
        frame=frame.drop(columns='first_anomaly_step').merge(first,on=keys,how='left')
        frame['first_anomaly_step']=frame.first_anomaly_step.fillna(0).astype(int)
        for event in ('nan','inf','inference_error','observation_bounds','divergence'):
            counts=event_frame[event_frame.event.eq(event)].drop_duplicates(keys+['horizon']).groupby(keys).size().rename(event+'_steps')
            frame=frame.drop(columns=event+'_steps').merge(counts,on=keys,how='left')
            frame[event+'_steps']=frame[event+'_steps'].fillna(0).astype(int)
    frame['flagged']=frame.first_anomaly_step>0
    frame.to_csv(output/'validity_by_origin.csv',index=False)
    summary=frame.groupby(['model','mode']).agg(requested_origins=('source_step','size'),
        failed_origins=('failed','sum'),flagged_origins=('flagged','sum'),
        requested_steps=('requested','sum'),attempted_steps=('attempted','sum'),finite_steps=('finite','sum'))
    for event in ('nan','inf','inference_error','observation_bounds','divergence'):
        summary[event+'_steps']=frame.groupby(['model','mode'])[event+'_steps'].sum()
    summary['failure_fraction']=summary.failed_origins/summary.requested_origins
    summary['finite_fraction']=summary.finite_steps/summary.requested_steps
    summary['observation_bounds_applicable']=True
    summary['physical_rate_check_applicable']=False
    summary.to_csv(output/'validity_summary.csv')


def main():
    torch.set_num_threads(2)
    config = yaml.safe_load((ROOT/'configs/experiments/week5_ib.yaml').read_text())
    for path, expected in config['sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen file changed: '+path)
    split = config['splits']
    if any(set(split[a]) & set(split[b]) for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Overlapping episodes')
    data = np.load(ROOT/config['data']); obs, actions = data['observations'], data['actions']
    horizon = config['max_horizon']
    output = ROOT/config['output']; output.mkdir(parents=True, exist_ok=True)
    metrics = []; logs = []; events = []; origin_rows = []
    for ep in split['test']:
        origin_rows.extend(dict(episode_id=ep, source_step=int(t)) for t in range(19,len(actions[ep])-horizon+1))
    pd.DataFrame(origin_rows).to_csv(output/'origins.csv', index=False)
    for spec in config['models']:
        model = None if spec['kind'] == 'ib_persistence' else load_dynamics(spec,ROOT,hashes=config['sha256'])
        saved = {}
        for mode in config['modes']:
            predictions = []; raw_values = []; masks = []; truths = []
            for ep in split['test']:
                starts = np.arange(19,len(actions[ep])-horizon+1)
                window = np.stack([np.concatenate([obs[ep,t-19:t+1], actions[ep,t-19:t+1]], axis=-1) for t in starts])
                p = np.full((len(starts),horizon,6),np.nan); raw = p.copy()
                attempted = np.zeros((len(starts),horizon),bool)
                actual = np.stack([obs[ep,starts+k+1] for k in range(horizon)],axis=1)
                active = np.ones(len(starts),bool); failure = np.zeros(len(starts),int)
                reason = ['']*len(starts)
                for k in range(horizon):
                    ids = np.flatnonzero(active)
                    if not len(ids): break
                    inputs = ib_window(obs[ep], actions[ep], starts[ids], k, mode, window[ids])
                    value, errors = predict_ib(model,inputs)
                    for j, origin in enumerate(ids):
                        attempted[origin,k] = True; raw[origin,k] = value[j]
                        common = dict(model=spec['name'],mode=mode,episode_id=ep,source_step=int(starts[origin]),horizon=k+1)
                        if errors[j]:
                            active[origin]=False; failure[origin]=k+1; reason[origin]=errors[j]
                            events.append(dict(**common,event='inference_error',variable='all',value=np.nan,magnitude=np.nan,detail=errors[j]))
                            continue
                        nonfinite = ~np.isfinite(value[j])
                        for index in np.flatnonzero(nonfinite):
                            events.append(dict(**common,event='nan' if np.isnan(value[j,index]) else 'inf',variable=config['variables'][index],value=value[j,index],magnitude=np.nan,detail=''))
                        if nonfinite.any():
                            active[origin]=False; failure[origin]=k+1; reason[origin]='nonfinite'; continue
                        p[origin,k]=value[j]
                        low, high = np.array(config['validity']['low']),np.array(config['validity']['high'])
                        for index in np.flatnonzero((value[j]<low)|(value[j]>high)):
                            events.append(dict(**common,event='observation_bounds',variable=config['variables'][index],value=value[j,index],
                                magnitude=float(max(low[index]-value[j,index],value[j,index]-high[index])),detail='flag only'))
                        if np.abs(value[j]).max()>config['validity']['extreme_abs']:
                            active[origin]=False; failure[origin]=k+1; reason[origin]='divergence'
                            events.append(dict(**common,event='divergence',variable='all',value=float(np.abs(value[j]).max()),magnitude=np.nan,detail='numerical threshold'))
                        if mode == 'free_running' and k+1<horizon:
                            window[origin,:-1]=window[origin,1:].copy()
                            window[origin,-1]=np.concatenate([value[j],actions[ep,starts[origin]+k+1]])
                predictions.append(p); raw_values.append(raw); masks.append(attempted); truths.append(actual)
                for i,t in enumerate(starts):
                    logs.append(dict(model=spec['name'],mode=mode,episode_id=ep,source_step=int(t),failure_step=int(failure[i]),reason=reason[i],
                                     requested=horizon,attempted=int(attempted[i].sum()),finite=int(np.isfinite(p[i]).all(axis=-1).sum())))
                for h in range(1,horizon+1):
                    for metric in rollout_errors(actual,p,h):
                        metrics.append(dict(model=spec['name'],family=spec.get('family','ib_persistence'),seed=spec.get('seed',-1),
                            mode=mode,condition='recorded_actions',episode_id=ep,horizon=h,unit=config['units'],
                            failed_origins=int(((failure>0)&(failure<=h)).sum()),**metric))
            saved[mode]=np.concatenate(predictions); saved[mode+'__raw']=np.concatenate(raw_values)
            saved[mode+'__attempted']=np.concatenate(masks); saved['actual']=np.concatenate(truths)
        np.savez_compressed(output/f'{spec["name"]}_predictions.npz',**saved)
        print('Completed '+spec['name'],flush=True)
    metrics=pd.DataFrame(metrics); summary,seeds=summarize_rollout_metrics(metrics)
    # IB 数值不是摄氏度；复用计算后立即重命名，禁止误标物理单位。
    for frame in (metrics,summary,seeds):
        frame.columns=[c.replace('_c','_native') if c.startswith(('mae_c','rmse_c')) else c for c in frame.columns]
    metrics.to_csv(output/'metrics_by_episode.csv',index=False); summary.to_csv(output/'metrics_by_horizon.csv',index=False)
    summary[summary.horizon.isin(config['horizons'])].to_csv(output/'metrics_key_horizons.csv',index=False)
    seeds.to_csv(output/'metrics_across_seeds.csv',index=False)
    pd.DataFrame(logs).to_csv(output/'failures.csv',index=False)
    pd.DataFrame(events,columns=['model','mode','episode_id','source_step','horizon','event','variable','value','magnitude','detail']).to_csv(output/'validity_events.csv.gz',index=False,compression='gzip')
    save_ib_validity(logs,events,output)
    (output/'protocol_snapshot.json').write_text(json.dumps(config,indent=2))
    from scripts.summarize_week5 import summarize_folder
    summarize_folder(output)


if __name__ == '__main__': main()
