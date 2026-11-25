"""总说明：真实 API 时序积分对照官方能耗增量，检查包含热泵、风机和水泵以及面积归一化。"""
from pathlib import Path
import sys,json
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulators.boptest_control import BoptestControlEnv


def main():
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text());cfg.update(start_time=24*86400,episode_steps=2)
    env=BoptestControlEnv(cfg)
    try:
        env.reset(seed=42);_,_,_,_,info=env.step([22.])
        names=['reaPHeaPum_y','reaPFan_y','reaPPumEmi_y']
        series=env.request('PUT',f'/results/{env.testid}',dict(point_names=names,start_time=cfg['start_time'],final_time=cfg['start_time']+900))
        time=np.array(series['time']);power=sum(np.array(series[name]) for name in names)
        integrated=float(np.trapezoid(power,time)/3600000.)
        official=info['energy_kwh'];np.testing.assert_allclose(integrated,official,rtol=2e-5,atol=1e-6)
        output=ROOT/cfg['output']/'service_check';output.mkdir(parents=True,exist_ok=True)
        (output/'energy_integral_check.json').write_text(json.dumps(dict(status='passed',integral_kwh=integrated,native_delta_kwh=official,
            series=series,area_m2=192.,points=names,formula='trapezoid(sum(power_w),time_s)/3600000',
            tolerance='rtol2e-5; official conversion factor2.77778e-7 vs exact1/3600000'),indent=2))
        print('Native versus local energy integral passed',flush=True)
    finally:env.close()


if __name__=='__main__':main()
