"""热工概率校准独立扩展：先仅验证集拟合并落盘，再固定参数评估测试集。"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import yaml
from models.dynamics_base import load_dynamics
from evaluators.rollout import rollout_episode
from metrics.calibration import fit_interval_scale,interval_arrays,interval_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config,root):
    """检查完整 episode 隔离和锁定资产；不修改模型、切分或既有主评测。"""
    protocol=config['evaluation']
    for path,expected in protocol['sha256'].items():
        if sha(root/path)!=expected: raise ValueError('Frozen asset changed: '+path)
    split=protocol['data']['split_ids']
    if any(set(split[a])&set(split[b]) for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Overlapping episode splits')
    return protocol


def predict_split(config,root,split):
    """各成员独立自由递推/teacher forcing，使用主协议共同起点和天气边界。"""
    protocol=config['evaluation']; output=root/config['output']/split
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(root/protocol['data']['directory']/f'{split}.csv',skipinitialspace=True)
    frame.columns=frame.columns.str.strip(); frame['episode_id']=frame.episode_id.str.strip()
    if set(frame.episode_id)!=set(protocol['data']['split_ids'][split]): raise ValueError('Split manifest mismatch')
    episodes=[]; origins=[]
    for episode_id in protocol['data']['split_ids'][split]:
        episode=frame[frame.episode_id.eq(episode_id)].sort_values('source_step')
        if not np.array_equal(episode.source_step,np.arange(len(episode))): raise ValueError('Noncontiguous episode')
        arrays=[np.stack(episode[column].map(json.loads)) for column in
            ('state','action_applied','exogenous_available_at_decision','next_state')]
        count=len(episode)-protocol['max_horizon']+1
        if count!=protocol['origins']['expected_origins_per_episode']: raise ValueError('Origin count mismatch')
        episodes.append((episode_id,arrays,json.loads(episode.iloc[0].environment_config)))
        origins.extend(dict(episode_id=episode_id,source_step=i) for i in range(count))
    pd.DataFrame(origins).to_csv(output/'origins.csv',index=False)
    families={}; failure_rows=[]
    for family in config['families']:
        specs=[spec for spec in protocol['models'] if spec['name'].startswith(family+'_seed')]
        if sorted(spec['seed'] for spec in specs)!=[42,43,44]: raise ValueError('Expected three fixed training seeds')
        family_values={}
        for spec in sorted(specs,key=lambda x:x['seed']):
            model=load_dynamics(spec,root,device=config['device'],hashes=protocol['sha256'])
            for condition in protocol['information']['conditions']:
                for mode in protocol['modes']:
                    values=[]; truths=[]
                    for episode_id,arrays,physics in episodes:
                        result=rollout_episode(model,*arrays,protocol['max_horizon'],mode,condition,
                            protocol['validity']['divergence_bounds_c'],protocol['validity'],physics)
                        values.append(result['predicted']); truths.append(result['actual'])
                        failure_rows.extend(dict(model=spec['name'],condition=condition,mode=mode,
                            episode_id=episode_id,source_step=i,failure_step=int(step),reason=result['failure_reason'][i])
                            for i,step in enumerate(result['failure_step']))
                    key=condition+'__'+mode
                    family_values.setdefault(key,[]).append(np.concatenate(values))
                    truth=np.concatenate(truths)
                    if 'actual' in family_values: np.testing.assert_array_equal(family_values['actual'],truth)
                    family_values['actual']=truth
            print(f'Completed {split}: {spec["name"]}',flush=True)
        values={key:np.stack(value) if key!='actual' else value for key,value in family_values.items()}
        np.savez_compressed(output/f'{family}_members.npz',**values)
        families[family]=values
    pd.DataFrame(failure_rows).to_csv(output/'failures.csv',index=False)
    return families,pd.DataFrame(origins)


def calibrate(config_path,root):
    """这一阶段只读取验证集目标；在任何测试评估前写出并锁定 q。"""
    config=yaml.safe_load(Path(config_path).read_text()); protocol=validate_config(config,root)
    torch.set_num_threads(2)
    families,origins=predict_split(config,root,'val')
    output=root/config['output']; rows=[]
    for family,data in families.items():
        for condition in protocol['information']['conditions']:
            for mode in protocol['modes']:
                key=condition+'__'+mode
                result=fit_interval_scale(data[key],data['actual'],config['nominal_coverage'],config['scale_floor_c'],split='val')
                for h in range(protocol['max_horizon']):
                    for index,variable in enumerate(('air','wall')):
                        rows.append(dict(family=family,condition=condition,mode=mode,horizon=h+1,variable=variable,
                            q=result['q'][h,index],requested=result['requested'],valid_scores=result['valid_scores'][h,index],
                            rank=result['rank'][h,index],status=result['status'][h,index],nominal=config['nominal_coverage']))
    table=pd.DataFrame(rows); table.to_csv(output/'calibration_parameters.csv',index=False)
    snapshot=dict(config=config,config_sha256=sha(config_path),calibration_parameters_sha256=sha(output/'calibration_parameters.csv'),
        calibration_split='val',calibration_episodes=origins.episode_id.nunique(),calibration_origins=len(origins),
        targets_used='validation targets only; no test targets used',
        caveat='Validation was previously used for model selection; overlapping origins are dependent. No strict exchangeability/coverage guarantee.',
        source_sha256={str(root.joinpath(name).relative_to(root)):sha(root/name) for name in
            ('metrics/calibration.py','evaluators/calibration.py','scripts/calibrate_week5.py',
             'models/action_mlp.py','models/dynamics_base.py','evaluators/rollout.py')})
    (output/'calibration_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    return table


def evaluate(config_path,root):
    """只读取冻结 q；测试集用于评价 coverage/width，不重新选择系数或尺度下限。"""
    config=yaml.safe_load(Path(config_path).read_text()); protocol=validate_config(config,root)
    output=root/config['output']; snapshot=json.loads((output/'calibration_snapshot.json').read_text())
    if snapshot['config_sha256']!=sha(config_path) or snapshot['calibration_parameters_sha256']!=sha(output/'calibration_parameters.csv'):
        raise ValueError('Calibration assets/config changed; refusing silent test-time refit')
    for relative,expected in snapshot['source_sha256'].items():
        if sha(root/relative)!=expected: raise ValueError('Calibration/prediction code changed: '+relative)
    parameters=pd.read_csv(output/'calibration_parameters.csv')
    torch.set_num_threads(2)
    families,origins=predict_split(config,root,'test')
    rows=[]
    for family,data in families.items():
        saved={'actual':data['actual']}
        for condition in protocol['information']['conditions']:
            for mode in protocol['modes']:
                key=condition+'__'+mode
                group=parameters[(parameters.family==family)&(parameters.condition==condition)&(parameters['mode']==mode)]
                q=group.pivot(index='horizon',columns='variable',values='q').reindex(index=range(1,protocol['max_horizon']+1),columns=['air','wall']).to_numpy()
                fitted=interval_arrays(data[key],q,config['scale_floor_c'])
                saved[key+'__mean']=fitted['mean']; saved[key+'__sd']=fitted['sd']; saved[key+'__scale']=fitted['scale']
                for method,coeff in [('uncalibrated_gaussian_reference',np.full_like(q,config['uncalibrated_z'])),('validation_calibrated',q)]:
                    intervals=interval_arrays(data[key],coeff,config['scale_floor_c'])
                    saved[key+'__'+method+'__lower']=intervals['lower']; saved[key+'__'+method+'__upper']=intervals['upper']
                    for episode_id in protocol['data']['split_ids']['test']:
                        indices=origins.episode_id.eq(episode_id).to_numpy()
                        for h in range(protocol['max_horizon']):
                            for index,variable in enumerate(('air','wall')):
                                result=interval_metrics(data['actual'][indices,h,index],intervals['lower'][indices,h,index],intervals['upper'][indices,h,index])
                                rows.append(dict(family=family,condition=condition,mode=mode,method=method,
                                    episode_id=episode_id,horizon=h+1,variable=variable,nominal=config['nominal_coverage'],**result))
        np.savez_compressed(output/'test'/f'{family}_intervals.npz',**saved)
    metrics=pd.DataFrame(rows)
    keys=['family','condition','mode','method','horizon','variable','nominal']
    summary=metrics.groupby(keys).agg(episodes=('episode_id','size'),requested=('requested','sum'),
        valid=('valid','sum'),invalid=('invalid','sum'),covered=('covered','sum'),
        valid_fraction=('valid_fraction','mean'),coverage_mean=('coverage_all','mean'),
        coverage_episode_sd=('coverage_all','std'),conditional_coverage_mean=('coverage_valid','mean'),
        mean_width_c=('mean_width_c','mean'),width_episode_sd=('mean_width_c','std')).reset_index()
    summary['coverage_gap']=summary.coverage_mean-summary.nominal
    metrics.to_csv(output/'interval_metrics_by_episode.csv',index=False)
    summary.to_csv(output/'interval_metrics_by_horizon.csv',index=False)
    summary[summary.horizon.isin(protocol['horizons'])].to_csv(output/'interval_metrics_key_horizons.csv',index=False)
    (output/'test_snapshot.json').write_text(json.dumps(dict(calibration_snapshot_sha256=sha(output/'calibration_snapshot.json'),
        applied_calibration_sha256=sha(output/'calibration_parameters.csv'),test_episodes=origins.episode_id.nunique(),test_origins=len(origins),
        interval='pointwise by horizon and variable; not simultaneous trajectory coverage',
        independent_unit='episode; fixed three-seed ensemble is one predictor, not three independent tests',
        widths='valid finite intervals only; coverage_all counts invalid intervals as not covered'),indent=2))
    return summary
