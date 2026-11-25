"""总说明：审计 BOPTEST 数据隔离、实际动作、区间能耗、检查点与预定闭环覆盖，不将缺失实验当完成。"""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text());base=ROOT/cfg['output'];out=base/'audit';out.mkdir(parents=True,exist_ok=True)
    extension=yaml.safe_load((ROOT/'configs/experiments/week6_boptest_extensions.yaml').read_text())
    spans=sorted((e['start_day']-7,e['start_day']+3,e['episode_id']) for e in cfg['episodes'])
    assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    manifest=json.loads((ROOT/cfg['asset_manifest']).read_text())
    for path,sha in manifest['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    data=[]
    for ep in cfg['episodes']:
        path=base/'data'/(ep['episode_id']+'.csv');meta=json.loads(path.with_suffix('.json').read_text());frame=pd.read_csv(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['csv_sha256']
        assert len(frame)==cfg['context_steps']+cfg['episode_steps']
        assert np.array_equal(frame.source_step,np.arange(len(frame)))
        assert frame.timestamp.diff().dropna().eq(cfg['step_seconds']).all()
        np.testing.assert_allclose(frame.action_requested_c,frame.action_applied_c,atol=1e-5)
        assert frame.energy_kwh.ge(0).all();states=np.stack(frame.state.map(json.loads));targets=np.stack(frame.next_state.map(json.loads))
        np.testing.assert_allclose(states[1:],targets[:-1],atol=1e-8)
        expected=frame.native_ener_tot_kwh_m2.iloc[-1]*192.-meta['initial_kpi']['ener_tot']*192.
        np.testing.assert_allclose(frame.energy_kwh.sum(),expected,atol=1e-7)
        assert frame.truncated.sum()==1 and not frame.terminated.any()
        data.append(dict(episode_id=ep['episode_id'],rows=len(frame),split=ep['split'],sha256=meta['csv_sha256']))
    models=json.loads((base/'training/models.json').read_text())
    for item in models['models']:assert hashlib.sha256((ROOT/item['checkpoint']).read_bytes()).hexdigest()==item['sha256']
    assert set(models['train_episodes']).isdisjoint(models['val_episodes'])
    for split in ('train','val'):
        assert hashlib.sha256((base/'data'/f'{split}.csv').read_bytes()).hexdigest()==models[split+'_sha256']
    expected=[]
    names=['embedded']+[m['name'] for m in models['models']]
    model_hashes={m['name']:m['sha256'] for m in models['models']};model_hashes['embedded']=None
    for e in cfg['episodes']:
        if e['split'] in ('train','val'):continue
        conditions=['clean']+(['noise_0p3','missing_4','goal_plus1'] if e['episode_id'] in extension['primary_reference_episodes'] else [])
        for condition in conditions:
            for name in names:expected.append(e['episode_id']+'__'+name+'__'+condition)
    missing=[];complete=[]
    for identity in expected:
        path=base/'closed_loop'/(identity+'.json')
        if not path.exists():missing.append(identity);continue
        record=json.loads(path.read_text());csv=path.with_suffix('.csv')
        assert record.get('checkpoint_sha256')==model_hashes[record['model']]
        assert record['status']=='complete' and record['steps']==cfg['episode_steps']
        assert hashlib.sha256(csv.read_bytes()).hexdigest()==record['csv_sha256']
        frame=pd.read_csv(csv);part=frame[~frame.context];assert len(part)==cfg['episode_steps']
        np.testing.assert_allclose(part.energy_kwh.sum(),record['energy_kwh'],atol=1e-7)
        np.testing.assert_allclose(part.cost.sum(),record['cost'],atol=1e-7)
        assert frame.iloc[-1].truncated and not frame.terminated.any()
        complete.append(identity)
    status=dict(status='complete' if not missing else 'in_progress',data_blocks=len(data),data_rows=sum(e['rows'] for e in data),models=len(models['models']),
        expected_controls=len(expected),completed_controls=len(complete),missing_controls=missing,
        note='18 deterministic calendar blocks; training seeds are not independent physical episodes; all fallback costs retained.')
    (out/'status.json').write_text(json.dumps(status,indent=2));pd.DataFrame(data).to_csv(out/'data_manifest.csv',index=False)
    # 文件级摘要使未提交代码和配置也能追溯；不采集环境变量或认证凭据。
    sources=[*ROOT.glob('scripts/*week6*.py'),*ROOT.glob('scripts/*week6*.sh'),ROOT/'simulators/boptest_control.py',ROOT/'models/boptest_dynamics.py',ROOT/'controllers/boptest_mpc.py',ROOT/'evaluators/robustness.py',*ROOT.glob('configs/experiments/week6*.yaml')]
    (out/'source_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps({k:v for k,v in status.items() if k!='missing_controls'}),flush=True)
    if args.require_complete and missing:raise RuntimeError(f'{len(missing)} frozen controls not yet complete')


if __name__=='__main__':main()
