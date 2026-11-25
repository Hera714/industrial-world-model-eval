# 总说明：采集回合、按回合分割、拟合线性动力学并比较预测与配对闭环。
import csv, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.collect_trajectories import collect_episodes
from models.system_identification import LinearDynamics
from simulators.simple_thermal import ThermalConfig
from controllers.mpc import MPCConfig
from evaluators.closed_loop import run_episode

OUT=Path('reports/week2/results/system_identification')
def known_next(x,u,d,c):
    air=x[...,0]; wall=x[...,1]; exchange=c.air_wall_conductance*(wall-air)
    return np.stack([air+c.dt_hours/c.air_capacity*(exchange+c.air_out_conductance*(d-air)+u+c.internal_load_kw),
       wall+c.dt_hours/c.wall_capacity*(-exchange+c.wall_out_conductance*(d-wall))],axis=-1)
def save_rows(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True); cfg=ThermalConfig(); mpc_cfg=MPCConfig()
    data=collect_episodes(45,1000,cfg)
    # 整个 episode 划分，验证和测试轨迹均不参与拟合。
    splits={'train':list(range(0,30)),'val':list(range(30,36)),'test':list(range(36,45))}
    for name,ids in splits.items(): data.loc[data.episode.isin(ids)].to_csv(OUT/f'{name}_trajectories.csv',index=False)
    tr=data[data.episode.isin(splits['train'])]; test=data[data.episode.isin(splits['test'])]
    model=LinearDynamics().fit(tr[['air','wall']].to_numpy(),tr.action.to_numpy(),tr.outdoor.to_numpy(),tr[['next_air','next_wall']].to_numpy())
    np.save(OUT/'linear_coefficients.npy',model.coef)
    (OUT/'parameters.json').write_text(json.dumps(model.parameters(),indent=2)+'\n',encoding='utf-8')
    # 离线自由 rollout：从真实状态出发，使用记录的未来动作和天气作为理想化开环输入；不重新读取未来状态。
    errors={'known':{h:[] for h in (1,5,10,20)},'learned':{h:[] for h in (1,5,10,20)}}
    for ep,frame in test.groupby('episode',sort=True):
        x=frame[['air','wall']].to_numpy(); y=frame[['next_air','next_wall']].to_numpy(); actions=frame.action.to_numpy(); weather=frame.outdoor.to_numpy()
        for start in range(len(frame)-20+1):
            states={'known':x[start].copy(),'learned':x[start].copy()}
            for j in range(20):
                states['known']=known_next(states['known'],actions[start+j],weather[start+j],cfg)
                states['learned']=model.predict_next(states['learned'],actions[start+j],weather[start+j])
                if j+1 in (1,5,10,20):
                    for key in states: errors[key][j+1].append(states[key]-y[start+j])
    prediction=[]
    for name,hs in errors.items():
        for h,arr in hs.items():
            arr=np.asarray(arr)
            for i,var in enumerate(('air','wall')):
                prediction.append({'model':name,'horizon':h,'variable':var,'count':len(arr),'mae_c':float(np.abs(arr[:,i]).mean()),'rmse_c':float(np.sqrt(np.mean(arr[:,i]**2)))})
    save_rows(OUT/'prediction_metrics.csv',prediction)
    # 新种子闭环：环境种子和 MPC 候选种子配对，但控制动作各自闭环产生。
    cases=[{'name':f'air{air}_seed{seed}','seed':seed,'initial_air_c':float(air),'initial_wall_c':float(wall)} for air,wall in ((18,19),(20,20),(22,21)) for seed in (42,43,44)]
    closed=[]
    for case in cases:
        paired={}
        for name,dynamics in [('known',None),('learned',model)]:
            rows,metrics,_=run_episode('mpc',case,cfg,mpc_cfg,dynamics=dynamics); paired[name]=rows
            save_rows(OUT/f"{case['name']}_{name}.csv",rows)
            closed.append({'scenario':case['name'],'model':name,'seed':case['seed'],'initial_air_c':case['initial_air_c'],**{k:v for k,v in metrics.items() if k!='fallback_reasons'},'fallback_reasons':json.dumps(metrics['fallback_reasons'])})
        for k in ('outdoor_used_c','process_air_c','process_wall_c','measurement_air_c','measurement_wall_c','measurement_outdoor_c'):
            np.testing.assert_array_equal([r[k] for r in paired['known']],[r[k] for r in paired['learned']])
    save_rows(OUT/'closed_loop_metrics.csv',closed)
    # 保存实验配置、集划分和统计，确保可复核。
    metadata={'episode_split':splits,'data_seed_start':1000,'control_cases':cases,'thermal_config':asdict(cfg),'mpc_config':asdict(mpc_cfg),
      'training':'真实状态、真实当前室外温度、实际动作及真实下一状态；内部热负荷恒定并吸收到截距',
      'offline_rollout':'测试 episode 真值初始化，使用记录的未来动作和真实天气做理想化开环评估；未来状态不作为输入',
      'control':'带噪状态与天气观测；每次重新规划，未来天气保持当前观测值；已知内部负荷；规划器候选种子固定',
      'caveat':'训练使用模拟器真实状态标签，实际场景未必可观测；成本是模拟评分非货币'}
    (OUT/'config.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ('known','learned'):
        rows=[r for r in prediction if r['model']==name and r['variable']=='air']
        axes[0].plot([r['horizon'] for r in rows],[r['rmse_c'] for r in rows],marker='o',label=name)
        axes[1].scatter([r['scenario'] for r in closed if r['model']==name],[r['cumulative_cost'] for r in closed if r['model']==name],label=name)
    axes[0].set(xlabel='Rollout horizon',ylabel='Air RMSE (°C)',title='Offline rollout'); axes[1].set(ylabel='Closed-loop score',title='Paired scenarios'); axes[1].tick_params(axis='x',rotation=60)
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'prediction_control.png',dpi=150); plt.close(fig)
    print('参数:',json.dumps(model.parameters(),ensure_ascii=False)); print(pd.DataFrame(prediction).to_string(index=False)); print(pd.DataFrame(closed).groupby('model')[['cumulative_cost','tracking_mae_c','energy_kwh','comfort_violation_rate','fallback_count']].agg(['mean','std']).to_string())
if __name__=='__main__': main()
