"""冻结评测输入与模型哈希，生成可直接读取的 YAML 配置。"""
from pathlib import Path
import json
import hashlib
import yaml
import pandas as pd

root=Path(__file__).resolve().parents[1]
digest=lambda path:hashlib.sha256((root/path).read_bytes()).hexdigest()
files={}
for path in ('data/processed/week4_thermal/manifest.json','data/processed/week4_thermal/trajectories.csv',
             'data/processed/week4_thermal/train.csv','data/processed/week4_thermal/val.csv','data/processed/week4_thermal/test.csv',
             'reports/week4/results/rollout/linear.npz'):
    files[path]=digest(path)
models=[dict(name='persistence',kind='persistence',checkpoint=None),
        dict(name='linear',kind='linear',checkpoint='reports/week4/results/rollout/linear.npz')]
for family,prefix in (('training','mlp'),('training_residual','residual')):
    config_path=f'reports/week4/results/{family}/config.json'
    cfg=json.loads((root/config_path).read_text())
    files[config_path]=digest(config_path)
    for run in cfg['runs']:
        path=run['checkpoint']
        files[path]=digest(path)
        models.append(dict(name=f"{prefix}_seed{run['seed']}",kind='action_mlp',checkpoint=path,seed=run['seed']))
manifest=json.loads((root/'data/processed/week4_thermal/manifest.json').read_text())
# 写入配置前检查实际文件与清单一致，保证每个起点支持完整时域。
seen=set()
for name, ids in manifest['splits'].items():
    frame=pd.read_csv(root/f'data/processed/week4_thermal/{name}.csv',skipinitialspace=True)
    frame.columns=frame.columns.str.strip()
    actual=set(frame.episode_id.str.strip())
    if actual != set(ids) or seen & actual:
        raise ValueError('Episode manifest mismatch or overlap')
    seen |= actual
    for _,episode in frame.groupby('episode_id'):
        if sorted(episode.source_step.tolist()) != list(range(144)):
            raise ValueError('Incomplete episode')
# 本步只冻结热工主评测；IB 采用独立信息组配置，实施时单独锁定模型。
config=dict(protocol_version='1.0',environment='simple_thermal',models=models,
    data=dict(directory='data/processed/week4_thermal',split_ids=manifest['splits']),
    sha256=files,training_seeds=[42,43,44],evaluation_seed=2026,
    horizons=[1,5,10,20,40,80],max_horizon=80,dt_hours=1/6,
    origins=dict(rule='all contiguous origins supporting full max_horizon in each episode',expected_test_episodes=18,expected_origins_per_episode=65),
    information=dict(state='true air and wall at origin; true current state every step only for teacher forcing',
        action='recorded applied action sequence',exogenous_order=['outdoor_c','load_kw'],
        conditions={'recorded_observation':'future recorded contemporaneous outdoor observations plus known load',
                    'persistence_forecast':'outdoor observation at origin held constant plus known load'}),
    modes=['teacher_forcing','free_running'],units='Celsius',
    validity=dict(nonfinite='stop at first NaN/Inf',inference_error='stop and log exception',
        observation_bounds_c=[None,None],plausibility_bounds_c=[-10,50],divergence_bounds_c=[-100,100],
        rate_check=dict(outdoor_bounds_c=[-5,17],power_bounds_kw=[0,5],load_kw=.3,noise_margin_c=.03),
        plausibility_action='flag and continue',rate_action='flag and continue',divergence_action='stop',
        automatic_projection=False),
    aggregation=dict(primary='equal-weight mean of per-episode metrics',dispersion='sample SD across episodes',
        model_seeds='report each seed then mean and sample SD; do not pool seeds with episodes',
        endpoints=['terminal','prefix'],failure_denominator='all requested origins',
        valid_error='finite available predictions only, report count and validity fraction alongside',
        confidence_intervals='none in protocol v1',selection='no checkpoint or threshold tuning on test'),
    output='reports/week5/results')
destination=root/'configs/experiments/week5_evaluation.yaml'
destination.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
print(destination)
