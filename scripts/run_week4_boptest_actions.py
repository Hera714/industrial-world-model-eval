"""通过官方 API 核对输入并采集覆盖温度设定值的配对响应。"""
from pathlib import Path
import sys
import json
import os
os.environ['no_proxy']='127.0.0.1,localhost'
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulators.boptest_adapter import BoptestEnv


def main():
    """相同预热和时间窗口，分别执行内置控制与温度设定值阶跃。"""
    cfg=yaml.safe_load((ROOT/'configs/environments/boptest.yaml').read_text())
    output=ROOT/'reports/week4/results/boptest_actions'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]; kpis={}
    for scenario in ('embedded','setpoint_step'):
        env=BoptestEnv(cfg)
        try:
            _,initial=env.reset(seed=42)
            inputs=env.request('GET',f'/inputs/{env.testid}')
            measurements=env.request('GET',f'/measurements/{env.testid}')
            (output/'inputs.json').write_text(json.dumps(inputs,indent=2))
            (output/'measurements.json').write_text(json.dumps(measurements,indent=2))
            print(json.dumps(inputs,indent=2),flush=True)
            assert inputs['oveTSet_u']['Unit']=='K'
            for step in range(96):
                # 前12小时设为20°C，后12小时22°C，覆盖保持一个900秒控制步。
                target=293.15 if step<48 else 295.15
                spec=inputs['oveTSet_u']
                assert spec['Minimum'] <= target <= spec['Maximum']
                command={'oveTSet_activate':0} if scenario=='embedded' else {'oveTSet_activate':1,'oveTSet_u':target}
                points=env.request('POST',f'/advance/{env.testid}',command)
                assert points['time']==cfg['start_time']+(step+1)*900
                actual=env.request('PUT',f'/results/{env.testid}',dict(point_names=['oveTSet_u','oveHeaPumY_u'],start_time=points['time'],final_time=points['time']))
                rows.append(dict(scenario=scenario,step=step+1,hour=(step+1)/4,
                    timestamp=points['time'],requested_setpoint_k=target if scenario!='embedded' else None,
                    activate=command['oveTSet_activate'],applied_setpoint_k=actual['oveTSet_u'][-1],
                    air_c=points['reaTZon_y']-273.15,heatpump_power_w=points['reaPHeaPum_y'],
                    compressor_signal=actual['oveHeaPumY_u'][-1],raw_measurements=json.dumps(points)))
                if scenario!='embedded':
                    np.testing.assert_allclose(actual['oveTSet_u'][-1],target,atol=1e-6)
            kpis[scenario]=env.request('GET',f'/kpi/{env.testid}')
        finally:
            env.close()
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'trajectories.csv',index=False)
    (output/'config_and_kpi.json').write_text(json.dumps(dict(config=cfg,kpis=kpis,
        protocol='paired deterministic weather/calendar and warmup; 20C to 22C setpoint at 12h'),indent=2))
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,constrained_layout=True)
    for scenario,part in frame.groupby('scenario'):
        axes[0].plot(part.hour,part.applied_setpoint_k-273.15,label=scenario)
        axes[1].plot(part.hour,part.air_c,label=scenario)
        axes[2].plot(part.hour,part.heatpump_power_w,label=scenario)
    axes[0].set_ylabel('Applied setpoint (°C)')
    axes[1].set_ylabel('Zone temperature (°C)')
    axes[2].set(ylabel='Heat pump input (W)',xlabel='Hours since evaluation start')
    for axis in axes:
        axis.legend(); axis.grid(alpha=.2)
    fig.savefig(output/'action_response.png',dpi=150)
    print(json.dumps(kpis,indent=2))


if __name__=='__main__':
    main()
