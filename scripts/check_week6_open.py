"""总说明：验收开环共同起点与完整矩阵，独立核对原始预测、真值、误差和有效比例。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]


def main():
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text())
    base=ROOT/cfg['output'];out=base/'open_loop'
    points=pd.read_csv(out/'endpoint_errors.csv.gz');models=json.loads((base/'training/models.json').read_text())['models']
    conditions=['ideal_weather_reference','persistence_weather','noise_0p1','noise_0p3','missing_1','missing_4']
    modes=['teacher_forcing','free_running'];variables=cfg['state_names']+['interval_energy']
    origins=[];inputs={};expected_keys=set()
    for e in cfg['episodes']:
        if e['split'] in ('train','val'):continue
        eid=e['episode_id'];frame=pd.read_csv(base/'data'/(eid+'.csv'))
        targets=np.column_stack([np.stack(frame.next_state.map(json.loads)),frame.energy_kwh])
        starts=list(range(cfg['context_steps'],len(frame)-max(cfg['horizons'])+1,4));inputs[eid]=(targets,starts)
        for index,start in enumerate(starts):
            origins.append(dict(episode_id=eid,origin_index=index,source_step=start,timestamp=int(frame.timestamp.iloc[start]),
                history_start_step=start-cfg['history_steps']+1,max_horizon=20,
                state=frame.state.iloc[start],actions=json.dumps(frame.action_applied_c.iloc[start:start+20].tolist()),
                weather=json.dumps(frame.exogenous.iloc[start:start+20].map(json.loads).tolist())))
        for m in models:
            for c in conditions:
                for mode in modes:expected_keys.add(f'{eid}__{m["name"]}__{c}__{mode}')
    pd.DataFrame(origins).to_csv(out/'origins.csv',index=False)
    raw=np.load(out/'raw_predictions.npz');assert set(raw.files)==expected_keys
    expected_rows=len(origins)*len(models)*len(conditions)*len(modes)*len(cfg['horizons'])*len(variables)
    assert len(points)==expected_rows
    identity=['episode_id','model','condition','mode','start','horizon','variable']
    assert not points.duplicated(identity).any()
    for (eid,model,condition,mode),part in points.groupby(['episode_id','model','condition','mode']):
        target,starts=inputs[eid];prediction=raw[f'{eid}__{model}__{condition}__{mode}']
        assert prediction.shape==(len(starts),20,6)
        assert set(part.start)==set(starts)
        for h in cfg['horizons']:
            for j,variable in enumerate(variables):
                rows=part[(part.horizon==h)&(part.variable==variable)].sort_values('start')
                assert len(rows)==len(starts)
                truth=target[np.asarray(starts)+h-1,j];pred=prediction[:,h-1,j]
                np.testing.assert_allclose(rows.actual,truth,rtol=1e-12,atol=1e-10)
                np.testing.assert_allclose(rows.prediction,pred,equal_nan=True,rtol=1e-12,atol=1e-10)
                np.testing.assert_allclose(rows.abs_error,np.abs(pred-truth),equal_nan=True,atol=1e-10)
                assert np.array_equal(rows.valid.to_numpy(),np.isfinite(pred))
    summary=pd.read_csv(out/'metrics.csv')
    np.testing.assert_allclose(summary.valid_fraction,summary.valid/summary.requested,atol=1e-12)
    result=dict(status='passed',models=8,test_time_blocks=8,origins=len(origins),origins_per_block=44,
        conditions=6,modes=2,horizons=cfg['horizons'],variables=6,endpoint_records=len(points),
        finite_endpoints=int(points.valid.sum()),invalid_endpoints=int((~points.valid).sum()),
        physical_warning_records=int(points.physical_warning.sum()),
        caveat='Physical warning is a trajectory-level flag repeated across variables; finite does not imply physically valid.',
        implementation='BOPTEST history/output-energy contract uses scripts/evaluate_week6_open.py rollout, not the thermal-specific evaluator.')
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
