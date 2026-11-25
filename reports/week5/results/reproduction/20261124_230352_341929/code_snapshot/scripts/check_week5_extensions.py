"""核验独立扩展与主评测的失败计数、GRU 首步一致性、历史范围和有效性汇总。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_week5_ib import save_ib_validity

for relative in ('reports/week5/results/rollout','reports/week5/results/projection/rollout','reports/week5/results/ib_history'):
    folder=ROOT/relative
    summary=pd.read_csv(folder/'metrics_by_horizon.csv')
    assert (summary.failed_origins<=summary.requested).all()
    assert (summary.unavailable_origins==summary.requested-summary.valid_origins).all()
    seeds=pd.read_csv(folder/'metrics_across_seeds.csv')
    assert (seeds.requested_model_origins==seeds.requested_per_run*seeds.runs).all()
    if 'ib_history' not in relative:
        validity=pd.read_csv(folder/'validity_summary.csv')
        validity['observation_bounds_applicable']=False
        validity.to_csv(folder/'validity_summary.csv',index=False)

folder=ROOT/'reports/week5/results/ib_history'
origins=pd.read_csv(folder/'origins.csv')
assert len(origins)==1010 and origins.source_step.min()==19 and origins.source_step.max()==220
for path in folder.glob('*_predictions.npz'):
    with np.load(path) as arrays:
        np.testing.assert_allclose(arrays['teacher_forcing'][:,0],arrays['free_running'][:,0],atol=1e-5)
        assert arrays['free_running'].shape==(1010,80,6)
events=pd.read_csv(folder/'validity_events.csv.gz')
logs=pd.read_csv(folder/'failures.csv')
logs['reason']=logs.reason.astype('string').fillna('')
save_ib_validity(logs.to_dict('records'),events.to_dict('records'),folder)
print('Passed explicit failure-count checks and IB 1010-origin / independent-window first-step checks.')
