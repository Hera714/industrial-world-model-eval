"""总说明：真实 HTTP 验证 BOPTEST 元数据、动作覆盖、释放、非法动作与 fallback，保存逐步证据。"""
from pathlib import Path
import sys,json,copy
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulators.boptest_control import BoptestControlEnv


def main():
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text())
    cfg.update(start_time=40*86400,episode_steps=5)
    out=ROOT/cfg['output']/'service_check';out.mkdir(parents=True,exist_ok=True)
    env=BoptestControlEnv(cfg);rows=[]
    try:
        obs,info=env.reset(seed=42)
        for name in ('inputs','measurements','forecast_points','scenario','version'):
            path='/version' if name=='version' else f'/{name}/{env.testid}'
            (out/f'{name}.json').write_text(json.dumps(env.request('GET',path),indent=2))
        for value in ([17.],[25.],[float('nan')]):
            try: env.step(value)
            except ValueError: pass
            else: raise AssertionError('Invalid action was accepted')
        assert env.steps==0
        for label,action,reason in [('override_20',[20.],''),('override_22',[22.],''),('release',None,''),('invalid_fallback',[float('nan')],''),('timeout_fallback',[21.],'timeout')]:
            result=env.step(None) if label=='release' else env.safe_step(action,reason)
            rows.append(dict(label=label,**result[-1]))
        assert rows[0]['override_active'] and rows[1]['override_active']
        assert not any(row['override_active'] for row in rows[2:])
        assert result[3] and not result[2]
        try: env.step([21.])
        except RuntimeError: pass
        else: raise AssertionError('Step after end accepted')
        (out/'real_api_check.json').write_text(json.dumps(dict(status='passed',config=cfg,rows=rows),indent=2))
        print('Real API override/release/fallback checks passed',flush=True)
    finally: env.close()


if __name__=='__main__': main()
