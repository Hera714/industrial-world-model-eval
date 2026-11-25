"""审计校准资产：验证参数只由验证集产生、固定应用测试集、结果有完整起点身份。"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from metrics.calibration import fit_interval_scale,interval_arrays

config=yaml.safe_load((ROOT/'configs/experiments/week5_calibration.yaml').read_text())
output=ROOT/config['output']; protocol=config['evaluation']
val=pd.read_csv(output/'val/origins.csv'); test=pd.read_csv(output/'test/origins.csv')
assert not set(val.episode_id)&set(test.episode_id)
assert len(val)==len(test)==1170
parameters=pd.read_csv(output/'calibration_parameters.csv')
assert len(parameters)==2*2*2*80*2 and parameters.status.eq('fitted').all()
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
snapshot=json.loads((output/'calibration_snapshot.json').read_text())
assert snapshot['calibration_parameters_sha256']==digest(output/'calibration_parameters.csv')
for family in config['families']:
    with np.load(output/'val'/f'{family}_members.npz') as validation, \
         np.load(output/'test'/f'{family}_members.npz') as testing, \
         np.load(output/'test'/f'{family}_intervals.npz') as intervals:
        for condition in protocol['information']['conditions']:
            for mode in protocol['modes']:
                key=condition+'__'+mode
                fitted=fit_interval_scale(validation[key],validation['actual'],config['nominal_coverage'],config['scale_floor_c'])
                selected=parameters[(parameters.family==family)&(parameters.condition==condition)&(parameters['mode']==mode)]
                q=selected.pivot(index='horizon',columns='variable',values='q').reindex(index=range(1,81),columns=['air','wall']).to_numpy()
                np.testing.assert_allclose(fitted['q'],q)
                applied=interval_arrays(testing[key],q,config['scale_floor_c'])
                for side in ('lower','upper'):
                    np.testing.assert_allclose(applied[side],intervals[key+'__validation_calibrated__'+side],equal_nan=True)
        # 校准扩展的点预测必须与已验收主评测一致，不因区间计算改变预测任务。
        main=np.load(ROOT/protocol['output']/'rollout'/f'{family}_seed42_predictions.npz')
        np.testing.assert_allclose(testing['recorded_observation__free_running'][0],main['recorded_observation__free_running'],atol=1e-5,equal_nan=True)
        main.close()
summary=pd.read_csv(output/'interval_metrics_by_horizon.csv')
assert summary.requested.eq(1170).all()
assert summary.episodes.eq(18).all()
assert (summary.valid+summary.invalid).eq(summary.requested).all()
files=[output/'calibration_parameters.csv',output/'calibration_snapshot.json',output/'test_snapshot.json',
       output/'interval_metrics_by_episode.csv',output/'interval_metrics_by_horizon.csv']
files.extend((output/'val').glob('*')); files.extend((output/'test').glob('*'))
manifest=dict(status='passed',calibration_split='val',validation_origins=len(val),test_origins=len(test),
    per_horizon_variable_calibrators=len(parameters),
    files_sha256={str(path.relative_to(ROOT)):digest(path) for path in files if path.is_file()},
    checks=['episode isolation','validation-only q reconstruction','test fixed-q reconstruction','main point prediction consistency','failure denominator'])
(output/'audit.json').write_text(json.dumps(manifest,indent=2))
print('Passed calibration audit: 1280 validation-only coefficients; test intervals fixed; 18+18 disjoint episodes.')
