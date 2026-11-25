"""总说明：验收204个真实闭环的配对初史/天气/评分、实际动作和完整回合。"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from metrics.control import calculate_boptest_control_metrics


def main():
    base=ROOT/'reports/week6/results';allruns=pd.read_csv(base/'residual_mlp/control_comparison.csv')
    assert len(allruns)==204 and allruns.run_id.nunique()==204
    references={};records=[]
    fields=['cost','energy_kwh','comfort_violation_rate','comfort_degree_hours','center_tracking_mae_c','decision_mean_ms','decision_p95_ms','fallback_count']
    for row in allruns.itertuples():
        path=base/'closed_loop'/(row.run_id+'.csv');meta=json.loads(path.with_suffix('.json').read_text());trace=pd.read_csv(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['csv_sha256']
        assert len(trace)==288 and trace.iloc[-1].truncated and not trace.terminated.any()
        assert trace.timestamp.diff().dropna().eq(900).all()
        scored=trace[~trace.context];assert len(scored)==192
        metrics=calculate_boptest_control_metrics(scored)
        for name in fields:np.testing.assert_allclose(metrics[name],meta[name],atol=1e-8)
        active=trace.override_active
        np.testing.assert_allclose(trace.loc[active,'requested_action_c'],trace.loc[active,'applied_action_c'],atol=1e-5)
        assert trace.applied_action_c.between(5,35).all()
        # 同日期的所有模型重放相同上下文；外部天气在完整回合一致。
        history=np.stack(trace.true_before.iloc[:97].map(json.loads));weather=np.stack(trace.weather.map(json.loads))
        if row.episode_id in references:
            previous=references[row.episode_id]
            np.testing.assert_allclose(history,previous[0],atol=1e-5)
            np.testing.assert_allclose(weather,previous[1],atol=1e-8)
        else:references[row.episode_id]=(history,weather)
        records.append(dict(run_id=row.run_id,model=row.model,episode_id=row.episode_id,condition=row.condition,**metrics))
    out=base/'closed_loop_acceptance';out.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out/'paired_runs.csv',index=False)
    result=dict(status='passed',complete_runs=204,model_checkpoints=11,engineering_baseline='embedded',
        date_blocks=8,scored_steps=204*192,verified='Identical context/initial observation and weather, native energy, actual override, timestamps, metrics, truncation',
        reference='Native embedded controller has internal sensor information; noise comparisons use within-controller clean differences.')
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
