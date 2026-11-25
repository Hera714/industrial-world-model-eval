"""总说明：独立核对动作计划、输出覆盖、冻结输入及失败计数，保存可追溯审核记录。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    """检查完整输出，防止缺失案例或改变来源后继续复用旧结果。"""
    config=yaml.safe_load((ROOT/'configs/experiments/week5_diagnostics.yaml').read_text())
    for relative,expected in (config['input_sha256']|config['evaluation']['sha256']).items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    output=ROOT/config['output']
    protocol=json.loads((output/'action/protocol.json').read_text())
    with np.load(output/'action/trajectories.npz') as data:
        assert data['true_state'].shape==(6,8,81,2)
        assert data['exogenous'].shape==(6,80,2)
        assert data['actions'].shape==(8,80)
        assert np.isfinite(data['true_state']).all()
        assert ((data['actions']>=0)&(data['actions']<=5)).all()
        assert len(protocol['model_names'])==8
    response=pd.read_csv(output/'action/response_metrics.csv')
    assert len(response)==672
    persistence=response[response.model.eq('persistence')]
    assert persistence.lag_error_minutes.isna().all()
    assert persistence.direction_accuracy.eq(0).all()
    points=pd.read_csv(output/'decomposition/endpoint_errors.csv.gz')
    assert len(points)==449280
    counts=pd.read_csv(output/'decomposition/case_counts.csv')
    assert counts.requested.eq(1170).all()
    assert int(counts.requested.sum())==len(points)
    assert int(counts.invalid_predictions.sum())==int((~points.valid_prediction).sum())
    cases=pd.read_csv(output/'decomposition/diagnostic_cases.csv.gz')
    expected=points.failed_by_h|~points.valid_prediction|points.trajectory_warning_by_h|points.large_error
    assert len(cases)==int(expected.sum())
    artifacts={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob('*')) if path.is_file() and path.name!='diagnostics_audit.json'}
    audit=dict(status='passed',endpoint_rows=len(points),flagged_rows=len(cases),
        numerical_failure_endpoints=int(counts.numerical_failures.sum()),artifact_sha256=artifacts)
    (output/'diagnostics_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in audit.items() if key!='artifact_sha256'}))


if __name__=='__main__':
    main()
