"""冻结 Week 5 扩展：原始/投影配对配置及 IB 独立信息组，不修改主协议。"""
from pathlib import Path
import json
import hashlib
import yaml

ROOT = Path(__file__).resolve().parents[1]
digest = lambda path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
base = yaml.safe_load((ROOT/'configs/experiments/week5_evaluation.yaml').read_text())
projection = dict(base)
projection['protocol_version'] = '1.0-projection-extension'
projection['models'] = list(base['models'])
for spec in base['models']:
    if spec['name'].startswith('mlp_seed'):
        projection['models'].append(dict(name='projected_'+spec['name'], kind='rate_projection',
            family='projected_mlp', seed=spec['seed'], base_model=spec,
            outdoor_bounds_c=[-5.,17.], noise_margin_c=.03))
projection['output'] = 'reports/week5/results/projection'
(ROOT/'configs/experiments/week5_projection.yaml').write_text(yaml.safe_dump(projection, sort_keys=False))

folder = Path('reports/week4/results/ib_history')
cfg = json.loads((ROOT/folder/'config.json').read_text())
models = [dict(name='ib_persistence', kind='ib_persistence')]
files = [str(folder/'episodes.npz'), str(folder/'config.json')]
for family in ('mlp','history_mlp','gru'):
    for seed in (42,43,44):
        path = str(folder/f'{family}_{seed}.pt')
        files.append(path)
        models.append(dict(name=f'ib_{family}_{seed}', kind='ib_'+family, checkpoint=path,
            family='ib_'+family, seed=seed, information_group='current' if family == 'mlp' else 'history20'))
ib = dict(protocol_version='1.0-ib-independent', environment='industrial_benchmark',
    data=str(folder/'episodes.npz'), splits=cfg['splits'], models=models,
    sha256={path:digest(path) for path in files}, horizons=[1,5,10,20,40,80], max_horizon=80,
    modes=['teacher_forcing','free_running'], units='IB native benchmark scale', history=20,
    origins='all steps 19..220 in each held-out 300-step episode; same origins for every model',
    condition='recorded actions; no future true observations in free running; no extra exogenous input',
    variables=['setpoint','velocity','gain','shift','fatigue','consumption'],
    validity=dict(low=[0]*6, high=[100,100,100,100,1000,1000], extreme_abs=1e6,
        observation_action='flag and continue', extreme_action='stop after retaining finite triggering point',
        rate='not applicable: no independently established physical rate bounds',
        extreme_note='heuristic numerical threshold; frozen for this extension, not a physical safety limit'),
    output='reports/week5/results/ib_history',
    aggregation='episode equal weight, then training-seed mean and sample SD; units never pooled',
    selection='use all prior checkpoints; no test-based tuning; new independent extension after thermal main experiment')
(ROOT/'configs/experiments/week5_ib.yaml').write_text(yaml.safe_dump(ib, sort_keys=False))
print('Frozen projection and independent IB extension configs')
