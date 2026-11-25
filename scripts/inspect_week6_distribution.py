"""总说明：只读BOPTEST实际训练数据，输出各字段训练范围、分位数与逐时间块范围外比例。"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]


def values(frame):
    """历史上下文属于模型输入支持，另标记有效目标以便核对统计分母。"""
    state=np.stack(frame.state.map(json.loads))
    weather=np.stack(frame.exogenous.map(json.loads))
    return np.column_stack([state,frame.action_applied_c,weather,frame.energy_kwh])


def main():
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text())
    base=ROOT/cfg['output'];out=base/'distribution';out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(base/'data/train.csv');array=values(train)
    names=cfg['state_names']+['setpoint_c','outdoor_c','solar_w_m2','interval_energy_kwh']
    units=['C','C','C','kW','kW','C','C','W/m2','kWh']
    ranges=[]
    for scope,mask in [('including_context',np.ones(len(train),bool)),('scored_targets',~train.context.to_numpy())]:
        for j,name in enumerate(names):
            v=array[mask,j];q=np.quantile(v,[0,.01,.05,.5,.95,.99,1])
            ranges.append(dict(scope=scope,variable=name,unit=units[j],count=len(v),mean=v.mean(),std=v.std(ddof=1),
                minimum=q[0],p01=q[1],p05=q[2],median=q[3],p95=q[4],p99=q[5],maximum=q[6]))
    pd.DataFrame(ranges).to_csv(out/'training_ranges.csv',index=False)
    rows=[];lo=array.min(0);hi=array.max(0)
    for episode in cfg['episodes']:
        frame=pd.read_csv(base/'data'/(episode['episode_id']+'.csv'));a=values(frame)[~frame.context.to_numpy()]
        for j,name in enumerate(names):
            rows.append(dict(episode_id=episode['episode_id'],split=episode['split'],start_day=episode['start_day'],policy=episode['policy'],
                variable=name,unit=units[j],minimum=a[:,j].min(),maximum=a[:,j].max(),mean=a[:,j].mean(),
                train_minimum=lo[j],train_maximum=hi[j],outside_training_fraction=np.mean((a[:,j]<lo[j])|(a[:,j]>hi[j]))))
    pd.DataFrame(rows).to_csv(out/'scenario_support.csv',index=False)
    scenes=json.loads((base/'protocol/scenario_manifest.json').read_text())
    (out/'scenario_parameters.json').write_text(json.dumps(scenes,indent=2))
    (out/'ood_definition.md').write_text('''# 任务2：训练分布与OOD判定

数据来自Week6实际BOPTEST训练块。training_ranges.csv同时报告包含上下文的输入支持和有效目标行的统计，含最小/最大、均值、标准差及分位数。scenario_support.csv只对各块有效目标时段计算支持范围外比例。范围外是逐变量边际证据，不能证明或否定联合分布变化。

主要参考日期160、292、340的室外温度均在训练支持内。日期172作为额外日历变化案例，温度范围外比例18.23%。暖季196、208分别为5.73%、62.50%，其中208提供明确的天气范围外推条件。冷季12、24温度均在训练范围内，归为季节保留工况。日期编号是全年仿真时间中的位置。

第二类OOD为同一参考日期的独立舒适上下界同时提高1°C，属于控制任务变化。设定值动作边界仍按执行配置固定，目标变化不等同于物理动力学改变。噪声/缺失另外作为观测鲁棒性条件。

这些统计使用已采集数据形成事后分布审计。原始场景配置和补充日期依据保留在week6_boptest_extensions.yaml，不声称全部场景均为严格事前注册。禁止根据测试误差重新选择日期。
''',encoding='utf-8')
    print(f'Written {len(ranges)} training-range rows and {len(rows)} scenario-support rows to {out}')


if __name__=='__main__':main()
