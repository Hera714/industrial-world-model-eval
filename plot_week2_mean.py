# 总说明：对三个已完成的配对场景逐时刻求均值，展示温度与动作的场景间波动。
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path('reports/week2/results/paired_comparison')
    names=('cold_seed42','mild_seed43','warm_seed44')
    curves={}
    for controller in ('rule','mpc'):
        runs=[]
        for name in names:
            with (root/f'{name}_{controller}_trajectory.csv').open(encoding='utf-8',newline='') as f:
                rows=list(csv.DictReader(f))
            times=np.array([float(r['hour']) for r in rows])
            air=np.array([float(r['true_air_after_c']) for r in rows])
            action=np.array([float(r['action_kw']) for r in rows])
            runs.append((times,air,action))
        # 检查场景具有相同时间网格，避免错位后求平均。
        for run in runs[1:]: np.testing.assert_allclose(run[0],runs[0][0],rtol=0,atol=1e-10)
        curves[controller]={'time':runs[0][0],'air':np.stack([r[1] for r in runs]),'action':np.stack([r[2] for r in runs])}
    np.testing.assert_allclose(curves['rule']['time'],curves['mpc']['time'],rtol=0,atol=1e-10)
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    with (root/'scenario_mean_trajectory.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['hour','controller','air_mean_c','air_std_c','action_mean_kw','action_std_kw'])
        w.writeheader()
        for controller,color in [('rule','tab:blue'),('mpc','tab:orange')]:
            c=curves[controller]; t=c['time']
            # 阴影为三个场景的总体标准差，用于描述波动，不作为置信区间。
            for i,key in enumerate(('air','action')):
                mean=c[key].mean(axis=0); sd=c[key].std(axis=0,ddof=0)
                axes[i].plot(t,mean,label=controller,color=color,drawstyle='steps-pre' if key=='action' else 'default')
                axes[i].fill_between(t,mean-sd,mean+sd,color=color,alpha=.14,step='pre' if key=='action' else None)
            am=c['air'].mean(0); ast=c['air'].std(0); um=c['action'].mean(0); ust=c['action'].std(0)
            for i in range(len(t)): w.writerow({'hour':t[i],'controller':controller,'air_mean_c':am[i],'air_std_c':ast[i],'action_mean_kw':um[i],'action_std_kw':ust[i]})
    axes[0].axhline(21,color='black',linestyle='--',label='Target 21 °C')
    axes[0].set(ylabel='True indoor temperature (°C)',title='Mean across three paired scenarios (shading: ±1 population SD)')
    axes[1].set(xlabel='Time (hours)',ylabel='Mean heating power (kW)')
    for ax in axes: ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(root/'scenario_mean_comparison.png',dpi=160); plt.close(fig)
    print('Saved',root/'scenario_mean_comparison.png',root/'scenario_mean_trajectory.csv')
if __name__=='__main__': main()
