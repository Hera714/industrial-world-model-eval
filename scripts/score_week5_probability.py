"""评分已有测试 ensemble：逐起点评分、episode/变量/horizon 统计及同回合配对。"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from metrics.probabilistic import empirical_crps_components
from evaluators.summary import summarize_episode_metrics,episode_seed_metrics,paired_episode_differences,summarize_pairs


def main():
    config_path=ROOT/'configs/experiments/week5_probability_scores.yaml'
    config=yaml.safe_load(config_path.read_text())
    for relative,expected in config['input_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen probability-score input changed: '+relative)
    source=ROOT/config['input_directory']; output=ROOT/config['output']
    origins=pd.read_csv(source/'origins.csv')
    if origins.duplicated(['episode_id','source_step']).any(): raise ValueError('Duplicate origin identity')
    failures=pd.read_csv(source/'failures.csv')
    output.mkdir(parents=True,exist_ok=True)
    (output/'origins.csv').write_bytes((source/'origins.csv').read_bytes())
    rows=[]
    for family in config['families']:
        saved={}
        with np.load(source/f'{family}_members.npz') as data:
            actual=data['actual']; saved['actual']=actual
            if actual.shape!=(len(origins),config['max_horizon'],2): raise ValueError('Unexpected target shape')
            for condition in config['conditions']:
                for mode in config['modes']:
                    key=condition+'__'+mode
                    members=data[key]
                    if members.shape!=(3,)+actual.shape: raise ValueError('Expected fixed 3-member empirical distribution')
                    components=empirical_crps_components(actual,members)
                    for name,value in components.items(): saved[key+'__'+name]=value
                    failure=failures[failures.model.str.startswith(family+'_seed')&failures.condition.eq(condition)&failures['mode'].eq(mode)].copy()
                    failure['failure_step']=failure.failure_step.replace(0,np.nan)
                    first=failure.groupby(['episode_id','source_step']).failure_step.min().reset_index()
                    first=origins.merge(first,on=['episode_id','source_step'],validate='one_to_one').failure_step.fillna(0).to_numpy()
                    score=components['empirical_crps_c']
                    for episode_id in origins.episode_id.unique():
                        ids=origins.episode_id.eq(episode_id).to_numpy()
                        for horizon in range(1,config['max_horizon']+1):
                            failed=int(((first[ids]>0)&(first[ids]<=horizon)).sum())
                            for index,variable in enumerate(config['variables']):
                                values=score[ids,:horizon,index]; finite=np.isfinite(values)
                                for definition in ('terminal','prefix_complete','prefix_available'):
                                    if definition=='terminal':
                                        selected=finite[:,-1]; points=values[selected,-1]
                                    elif definition=='prefix_complete':
                                        selected=finite.all(axis=1); points=values[selected].ravel()
                                    else:
                                        selected=finite.any(axis=1); points=values[finite]
                                    rows.append(dict(model=family+'_empirical3',family=family,seed=-1,
                                        mode=mode,condition=condition,horizon=horizon,definition=definition,
                                        variable=variable,unit=config['unit'],episode_id=episode_id,
                                        requested=int(ids.sum()),valid_origins=int(selected.sum()),
                                        complete_origins=int(finite.all(axis=1).sum()),valid_points=len(points),
                                        failed_origins=failed,empirical_crps_c=float(points.mean()) if len(points) else np.nan,
                                        member_count=3,distribution='equal_weight_empirical3'))
        np.savez_compressed(output/f'{family}_scores.npz',**saved)
    episodes=pd.DataFrame(rows); summary,ensemble_runs=summarize_episode_metrics(episodes)
    pairs=paired_episode_differences(episode_seed_metrics(episodes),{family:'thermal_full_state' for family in config['families']})
    pair_summary=summarize_pairs(pairs)
    episodes.to_csv(output/'scores_by_episode.csv',index=False)
    summary.to_csv(output/'scores_by_horizon.csv',index=False)
    summary[summary.horizon.isin(config['horizons'])].to_csv(output/'scores_key_horizons.csv',index=False)
    ensemble_runs.to_csv(output/'ensemble_run_summary.csv',index=False)
    pairs.to_csv(output/'paired_episode_scores.csv.gz',index=False,compression='gzip')
    pair_summary.to_csv(output/'paired_score_summary.csv',index=False)
    (output/'modeling_assumptions.json').write_text(json.dumps(dict(config=config,
        crps='CRPS of original finite empirical distribution, not of validation-calibrated intervals',
        nll=config['nll'],member_dependence='Different training seeds on shared data, not IID posterior draws',
        units='CRPS uses temperature units Celsius, not Celsius squared',
        interpretation='marginal one-variable scores, not joint multivariate or trajectory distribution scores',
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT/'metrics/probabilistic.py',Path(__file__),ROOT/'evaluators/summary.py')}),indent=2))
    print(summary.query('mode == "free_running" and definition == "terminal" and horizon in [1,20,80]')
        [['family','condition','variable','horizon','empirical_crps_c_mean','empirical_crps_c_episode_sd','valid_fraction']].to_string(index=False))


if __name__=='__main__': main()
