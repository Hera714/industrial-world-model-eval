"""总说明：按冻结时间块采集 BOPTEST 动作轨迹，保留上下文、原始测量、实际动作与官方步能耗。"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulators.boptest_control import BoptestControlEnv,measurement_state


def collect(cfg,episode):
    """每块独立初始化；故障不产生伪造完整回合，允许重启时跳过已经校验的完整文件。"""
    output=ROOT/cfg['output']/'data';output.mkdir(parents=True,exist_ok=True)
    path=output/(episode['episode_id']+'.csv'); meta=path.with_suffix('.json')
    total=cfg['context_steps']+cfg['episode_steps']
    if path.exists() and meta.exists():
        record=json.loads(meta.read_text())
        if record.get('status')=='complete' and record['rows']==total and record['csv_sha256']==hashlib.sha256(path.read_bytes()).hexdigest():
            print('skip '+episode['episode_id'],flush=True);return
    local=dict(cfg,start_time=episode['start_day']*86400,episode_steps=total)
    env=BoptestControlEnv(local);rows=[];rng=np.random.default_rng(episode['seed']);held=21.
    try:
        _,initial=env.reset(seed=episode['seed']);points=initial['measurements'];state=measurement_state(points)
        forecast=env.request('PUT',f'/forecast/{env.testid}',dict(point_names=cfg['exogenous'],horizon=total*cfg['step_seconds'],interval=cfg['step_seconds']))
        (output/(episode['episode_id']+'_forecast.json')).write_text(json.dumps(forecast))
        for k in range(total):
            # 每小时改变设定值；随机覆盖与预定日程覆盖同一合法区间，避免15分钟设备抖动。
            if k%4==0:
                held=float(rng.choice([18.,19.,20.,21.,22.,23.,24.])) if episode['policy']=='random_hold' else float([20.,22.,19.,23.,21.,24.][(k//16)%6])
            before=points; start=time.perf_counter(); _,_,term,trunc,info=env.step([held]);points=info['measurements']
            after=measurement_state(points)
            timestamp=local['start_time']+k*cfg['step_seconds']
            d=[float(np.interp(timestamp,forecast['time'],forecast[name])) for name in cfg['exogenous']]
            d[0]-=273.15
            rows.append(dict(episode_id=episode['episode_id'],split=episode['split'],source_step=k,
                timestamp=timestamp,seed=episode['seed'],policy=episode['policy'],context=k<cfg['context_steps'],
                state=json.dumps(state.tolist()),action_requested_c=held,action_applied_c=info['applied_action_c'],
                exogenous=json.dumps(d),next_state=json.dumps(after.tolist()),energy_kwh=info['energy_kwh'],
                native_ener_tot_kwh_m2=info['native_kpi']['ener_tot'],native_kpi=json.dumps(info['native_kpi']),
                raw_before=json.dumps(before),raw_after=json.dumps(points),override_active=info['override_active'],
                fallback_reason=info['fallback_reason'],api_elapsed_s=time.perf_counter()-start,terminated=term,truncated=trunc))
            state=after
            if k%96==95: print(f'{episode["episode_id"]}: {k+1}/{total}',flush=True)
        assert len(rows)==total and rows[-1]['truncated'] and not rows[-1]['terminated']
        frame=pd.DataFrame(rows);temp=path.with_suffix('.partial.csv');frame.to_csv(temp,index=False);temp.replace(path)
        meta.write_text(json.dumps(dict(status='complete',rows=len(rows),episode=episode,config=local,
            csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),initial=initial,
            initial_kpi=env.initial_kpi,final_kpi=rows[-1]['native_kpi'],
            exogenous_information='Exact API weather forecast at reset, used for diagnostics; persistence-weather mode uses only origin weather.'),indent=2))
    except Exception as exc:
        if rows: pd.DataFrame(rows).to_csv(path.with_suffix('.partial.csv'),index=False)
        (output/(episode['episode_id']+'_failure.json')).write_text(json.dumps(dict(error=repr(exc),completed=len(rows))))
        raise
    finally: env.close()


def main():
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text())
    manifest=json.loads((ROOT/cfg['asset_manifest']).read_text())
    for relative,expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    for episode in cfg['episodes']: collect(cfg,episode)
    output=ROOT/cfg['output']/'data'
    frames=[pd.read_csv(output/(e['episode_id']+'.csv')) for e in cfg['episodes']]
    for split in sorted({e['split'] for e in cfg['episodes']}):
        pd.concat([f for f in frames if f['split'].iloc[0]==split]).to_csv(output/(split+'.csv'),index=False)
    print(f'All {len(cfg["episodes"])} time blocks collected',flush=True)


if __name__=='__main__': main()
