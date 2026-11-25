"""用独立双重求和公式核验保存的经验 CRPS，并检查分布声明、分母和追溯关系。"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
config_path=ROOT/'configs/experiments/week5_probability_scores.yaml'
config=yaml.safe_load(config_path.read_text())
source=ROOT/config['input_directory']; output=ROOT/config['output']
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
for relative,expected in config['input_sha256'].items():
    if digest(ROOT/relative)!=expected: raise ValueError('Frozen scoring input changed: '+relative)
assert config['member_seeds']==[42,43,44] and not config['fair_iid_correction']
np.testing.assert_allclose(config['member_weights'],np.full(3,1/3))
origins=pd.read_csv(output/'origins.csv')
assert len(origins)==1170 and origins.episode_id.nunique()==18
for family in config['families']:
    with np.load(source/f'{family}_members.npz') as members_data,np.load(output/f'{family}_scores.npz') as scores:
        for condition in config['conditions']:
            for mode in config['modes']:
                key=condition+'__'+mode
                members=members_data[key]; target=members_data['actual']; count=len(members)
                finite=np.isfinite(members).all(axis=0)
                with np.errstate(invalid='ignore'):
                    first=np.abs(members-target[None]).mean(axis=0)
                    second=sum(np.abs(members[i]-members[j]) for i in range(count) for j in range(count))/(2*count*count)
                expected=np.where(finite,first-second,np.nan)
                np.testing.assert_allclose(scores[key+'__empirical_crps_c'],expected,atol=1e-12,equal_nan=True)
                assert scores[key+'__empirical_crps_c'].shape==(1170,80,2)
summary=pd.read_csv(output/'scores_by_horizon.csv')
assert summary.requested.eq(1170).all()
assert summary.episodes.eq(18).all()
assert summary[['empirical_crps_c_mean']].ge(0).all().all()
assert not any(column.lower().startswith('nll') for column in summary)
assumptions=json.loads((output/'modeling_assumptions.json').read_text())
assert assumptions['nll']['status']=='not_computed'
files=[path for path in output.iterdir() if path.is_file() and path.name!='score_audit.json']
audit=dict(status='passed',reference_formula='mean absolute target distance - all ordered pair distances/(2M^2)',
    empirical_member_count=3,model_families=config['families'],episodes=18,origins=1170,
    config_sha256=digest(config_path),nll='not computed: no continuous density declared',
    artifact_sha256={str(path.relative_to(ROOT)):digest(path) for path in files})
(output/'score_audit.json').write_text(json.dumps(audit,indent=2))
print('Passed empirical CRPS audit: all points match the independent M^2 pair-sum formula; no fabricated NLL.')
