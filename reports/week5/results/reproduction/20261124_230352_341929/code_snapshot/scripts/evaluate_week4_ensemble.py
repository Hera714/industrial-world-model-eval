"""固定切分下三种子 ensemble，计算均值预测、模型分歧与误差关系。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'reports/week4/results/rollout'
output=ROOT/'reports/week4/results/ensemble'
output.mkdir(parents=True,exist_ok=True)
origins=pd.read_csv(source/'origins.csv')
summaries=[]
bins=[]
fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
for row_index,family in enumerate(('mlp','residual')):
    members=[]
    actual=None
    for seed in (42,43,44):
        with np.load(source/f'{family}_seed{seed}_predictions.npz') as data:
            if actual is not None:
                np.testing.assert_array_equal(actual,data['actual'])
            actual=data['actual'].copy()
            members.append(data['free'].copy())
    members=np.stack(members)
    mean=members.mean(axis=0)
    std=members.std(axis=0,ddof=1)
    absolute=np.abs(mean-actual)
    np.savez_compressed(output/f'{family}.npz',actual=actual,mean=mean,std=std,members=members)
    # 固定展示清单中第一个起点；所有模型使用相同窗口。
    example=0
    minutes=np.arange(1,21)*10
    for index,variable in enumerate(('air','wall')):
        axis=axes[row_index,index]
        axis.plot(minutes,actual[example,:,index],color='black',label='Actual')
        axis.plot(minutes,mean[example,:,index],label='Ensemble mean')
        axis.fill_between(minutes,mean[example,:,index]-std[example,:,index],
            mean[example,:,index]+std[example,:,index],alpha=.25,label='±1 model SD')
        axis.set(xlabel='Minutes ahead',ylabel='Temperature (°C)',title=f'{family}: {variable}')
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
        for h in (1,5,10,20):
            error=absolute[:,h-1,index]
            spread=std[:,h-1,index]
            rho=float(spearmanr(spread,error).statistic)
            summaries.append(dict(family=family,variable=variable,horizon=h,
                origins=len(actual),mae_c=float(error.mean()),
                rmse_c=float(np.sqrt(np.mean((mean[:,h-1,index]-actual[:,h-1,index])**2))),
                mean_model_sd_c=float(spread.mean()),spearman=rho))
    # 固定第20步室温，按分歧排序分箱，避免不同预测时域混合造成相关性。
    spread=std[:,19,0]
    error=absolute[:,19,0]
    for indices in np.array_split(np.argsort(spread),10):
        bins.append(dict(family=family,count=len(indices),spread_c=float(spread[indices].mean()),
            mae_c=float(error[indices].mean())))
    axis=axes[row_index,2]
    axis.scatter(spread,error,s=4,alpha=.12)
    group=pd.DataFrame(bins).query('family == @family')
    axis.plot(group.spread_c,group.mae_c,'o-',color='red',label='10 equal-count bins')
    rho=spearmanr(spread,error).statistic
    axis.set(xlabel='Model disagreement SD (°C)',ylabel='Absolute ensemble error (°C)',
        title=f'{family}: air at step 20, Spearman={rho:.3f}')
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
fig.savefig(output/'ensemble_analysis.png',dpi=160)
plt.close(fig)
pd.DataFrame(summaries).to_csv(output/'metrics.csv',index=False)
pd.DataFrame(bins).to_csv(output/'disagreement_bins.csv',index=False)
(output/'protocol.json').write_text(json.dumps(dict(seeds=[42,43,44],
    sd='sample standard deviation across 3 models, ddof=1',
    rollout='each member recursively rolls out its own state, then aggregate',
    example=origins.iloc[0].to_dict(),correlation='descriptive Spearman across overlapping origins at fixed horizon',
    interval='mean ± model SD, uncalibrated disagreement band'),indent=2))
print(pd.DataFrame(summaries).query('horizon == 20').to_string(index=False))
