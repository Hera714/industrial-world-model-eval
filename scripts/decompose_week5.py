"""冻结测试误差分解：逐变量/功率/策略/初温/horizon，失败与大误差案例可追溯。"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluators.summary import summarize_episode_metrics


def endpoint_metadata(frame,origins,horizon):
    """功率取产生目标状态的那次转移；回合初温不等于窗口起点温度。"""
    rows=[]
    grouped={name:group.sort_values('source_step').reset_index(drop=True) for name,group in frame.groupby('episode_id')}
    for index,origin in enumerate(origins.itertuples()):
        episode=grouped[origin.episode_id]; start=int(origin.source_step); end=start+horizon-1
        if end>=len(episode) or not np.array_equal(episode.source_step,np.arange(len(episode))):
            raise ValueError('Invalid origin or noncontiguous episode')
        initial=json.loads(episode.iloc[0].state); current=episode.iloc[start]; target=episode.iloc[end]
        powers=episode.iloc[start:end+1].action_applied.map(lambda value:json.loads(value)[0]).to_numpy()
        power=float(powers[-1]); air=float(initial[0])
        rows.append(dict(origin_index=index,episode_id=origin.episode_id,source_step=start,
            target_transition_step=end,target_state_step=end+1,horizon=horizon,policy=current.policy,
            initial_air_c=air,initial_air_bin='<19' if air<19 else ('[19,21)' if air<21 else '>=21'),
            origin_power_kw=float(powers[0]),target_power_kw=power,mean_power_prefix_kw=float(powers.mean()),
            target_power_bin='<1' if power<1 else ('[1,4)' if power<4 else '>=4')))
    return pd.DataFrame(rows)


def group_errors(records):
    """先在 episode 内聚合绝对/平方误差，再复用统一的 episode/模型种子统计。"""
    pieces=[]
    for grouping,column in [('all',None),('power','target_power_bin'),('policy','policy'),('initial_air','initial_air_bin')]:
        frame=records.copy()
        frame['grouping']=grouping; frame['group']='all' if column is None else frame[column]
        keys=['model','family','seed','mode','condition','horizon','variable','episode_id','grouping','group']
        result=frame.groupby(keys,dropna=False).agg(requested=('origin_index','size'),
            valid_origins=('valid_prediction','sum'),complete_origins=('complete_prefix','sum'),
            failed_origins=('failed_by_h','sum'),mae_c=('abs_error_c','mean'),mse_c=('squared_error_c','mean')).reset_index()
        result['rmse_c']=np.sqrt(result.pop('mse_c')); result['valid_points']=result.valid_origins
        result['unit']='Celsius'; result['definition']='terminal'; pieces.append(result)
    episodes=pd.concat(pieces,ignore_index=True)
    summary,seeds=summarize_episode_metrics(episodes,context=['mode','condition','horizon','definition','variable','grouping','group'])
    return episodes,summary,seeds


def main():
    config=yaml.safe_load((ROOT/'configs/experiments/week5_diagnostics.yaml').read_text())
    protocol=config['evaluation']; settings=config['decomposition']
    for relative,expected in (config['input_sha256']|protocol['sha256']).items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected: raise ValueError('Frozen input changed: '+relative)
    source=ROOT/protocol['output']/'rollout'; output=ROOT/config['output']/'decomposition'
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(ROOT/protocol['data']['directory']/'test.csv',skipinitialspace=True)
    frame.columns=frame.columns.str.strip(); frame['episode_id']=frame.episode_id.str.strip()
    origins=pd.read_csv(source/'origins.csv')
    metadata={h:endpoint_metadata(frame,origins,h) for h in settings['horizons']}
    validity=pd.read_csv(source/'validity_by_origin.csv',keep_default_na=False)
    batches=[]
    for spec in protocol['models']:
        name=spec['name']; family='residual_mlp' if name.startswith('residual_') else spec['kind']
        with np.load(source/f'{name}_predictions.npz') as data:
            actual=data['actual']
            for condition in protocol['information']['conditions']:
                for mode in protocol['modes']:
                    prediction=data[condition+'__'+mode]
                    flags=validity[validity.model.eq(name)&validity.condition.eq(condition)&validity['mode'].eq(mode)]
                    flags=origins.merge(flags,on=['episode_id','source_step'],validate='one_to_one')
                    if len(flags)!=len(origins): raise ValueError('Missing validity identities')
                    for horizon,meta in metadata.items():
                        for index,variable in enumerate(('air','wall')):
                            rows=meta.copy(); a=actual[:,horizon-1,index]; p=prediction[:,horizon-1,index]
                            valid=np.isfinite(p)
                            rows['model']=name; rows['family']=family; rows['seed']=spec.get('seed',-1)
                            rows['mode']=mode; rows['condition']=condition; rows['variable']=variable
                            rows['actual_c']=a; rows['predicted_c']=p; rows['valid_prediction']=valid
                            rows['signed_error_c']=np.where(valid,p-a,np.nan)
                            rows['abs_error_c']=rows.signed_error_c.abs(); rows['squared_error_c']=rows.signed_error_c**2
                            rows['complete_prefix']=np.isfinite(prediction[:,:horizon,index]).all(axis=1)
                            rows['first_failure_step']=flags.first_failure_step.to_numpy()
                            rows['first_anomaly_step']=flags.first_anomaly_step.to_numpy()
                            rows['failure_reason']=flags.failure_reason.to_numpy()
                            rows['failed_by_h']=(rows.first_failure_step>0)&(rows.first_failure_step<=horizon)
                            rows['trajectory_warning_by_h']=(rows.first_anomaly_step>0)&(rows.first_anomaly_step<=horizon)
                            rows['large_error']=valid&(rows.abs_error_c>=settings['large_abs_error_c'])
                            batches.append(rows)
        print('Decomposed '+name,flush=True)
    records=pd.concat(batches,ignore_index=True)
    records.to_csv(output/'endpoint_errors.csv.gz',index=False,compression='gzip')
    episodes,summary,seeds=group_errors(records)
    episodes.to_csv(output/'group_metrics_by_episode.csv',index=False)
    summary.to_csv(output/'group_metrics.csv',index=False)
    seeds.to_csv(output/'group_metrics_across_seeds.csv',index=False)
    flagged=records.failed_by_h|~records.valid_prediction|records.trajectory_warning_by_h|records.large_error
    records[flagged].to_csv(output/'diagnostic_cases.csv.gz',index=False,compression='gzip')
    keys=['model','family','seed','mode','condition','horizon','variable']
    worst=records.sort_values(['failed_by_h','abs_error_c'],ascending=[False,False],na_position='first').groupby(keys,dropna=False,sort=False).head(1)
    worst.to_csv(output/'worst_cases.csv',index=False)
    records.groupby(keys).agg(requested=('origin_index','size'),numerical_failures=('failed_by_h','sum'),
        invalid_predictions=('valid_prediction',lambda value:(~value).sum()),
        large_error_endpoints=('large_error','sum'),trajectory_warning_endpoints=('trajectory_warning_by_h','sum')).reset_index().to_csv(output/'case_counts.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(config=config,
        input='original frozen test predictions; no new training or parameter selection',
        interpretation='group associations, not causal effect of power/policy; future target power used only for diagnostic labels',
        cases='Numerical failure, invalid output, trajectory warning and large error are separate boolean flags',
        all_cases=len(records),flagged_cases=int(flagged.sum()),worst_cases=len(worst),
        first_anomaly='trajectory-level flag, not necessarily an anomaly of the selected endpoint variable'),indent=2))
    print(f'Saved {len(records)} endpoints; {flagged.sum()} flagged; {len(worst)} deterministic worst-case representatives')


if __name__=='__main__': main()
