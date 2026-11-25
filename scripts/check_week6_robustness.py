"""总说明：重放固定故障随机流并逐值核对已执行轨迹，导出逐通道原始/污染/填补/缺失证据。"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluators.robustness import corrupt_boptest_observation


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--include-residual",action="store_true");args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text())
    base=ROOT/cfg['output'];output=base/'robustness';output.mkdir(parents=True,exist_ok=True)
    records=[];matrix=[]
    allowed={m['name'] for m in json.loads((base/'training/models.json').read_text())['models']}|{'embedded'}
    if args.include_residual:
        allowed|={m['name'] for m in json.loads((base/'residual_mlp/models.json').read_text())['models']}
        output=base/'residual_mlp/robustness';output.mkdir(exist_ok=True)
    for path in sorted((base/'closed_loop').glob('*.csv')):
        if path.name.endswith('.partial.csv'):continue
        frame=pd.read_csv(path)
        if "true_before" not in frame.columns:continue
        condition=frame.condition.iloc[0]
        if frame.model.iloc[0] not in allowed:continue
        if condition not in ('noise_0p3','missing_4'):continue
        eid=frame.episode_id.iloc[0];episode=next(e for e in cfg['episodes'] if e['episode_id']==eid)
        rng=np.random.default_rng(64042+episode['start_day']);last=None
        for row in frame.itertuples():
            original=np.asarray(json.loads(row.true_before));observed=np.asarray(json.loads(row.observed_before))
            raw,filled,mask,last=corrupt_boptest_observation(original,condition,rng,last,int(row.step)-cfg['context_steps'])
            np.testing.assert_allclose(filled,observed,atol=1e-10,rtol=0)
            assert bool(mask[1])==bool(row.missing_supply)
            for index,name in enumerate(cfg['state_names']):
                records.append(dict(run_id=row.run_id,episode_id=eid,model=row.model,condition=condition,step=row.step,
                    timestamp=row.timestamp,context=row.context,variable=name,original=original[index],
                    corrupted_before_fill=raw[index],controller_observation=filled[index],missing=mask[index],
                    additive_noise=raw[index]-original[index] if not mask[index] else np.nan))
        scored=frame[~frame.context]
        matrix.append(dict(run_id=frame.run_id.iloc[0],episode_id=eid,model=frame.model.iloc[0],condition=condition,
            noise_std_c=.3 if condition=='noise_0p3' else 0,missing_burst_steps=4 if condition=='missing_4' else 0,
            repeat_period_steps=24 if condition=='missing_4' else 0,rng_seed=64042+episode['start_day'],
            scored_steps=len(scored),missing_steps=int(scored.missing_supply.sum()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(matrix)==(72 if args.include_residual else 54)
    pd.DataFrame(records).to_csv(output/'observation_faults.csv.gz',index=False)
    pd.DataFrame(matrix).to_csv(output/'closed_loop_matrix.csv',index=False)
    result=dict(status='passed',verified_runs=len(matrix),channel_records=len(records),
        check='Exact replay of fixed RNG and causal fill equals every saved controller observation; missing masks equal saved flags',
        open_loop='Existing severity0/.1/.3C and bursts0/1/4steps results remain in open_loop; this audit covers closed-loop original observation logs.')
    (output/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
