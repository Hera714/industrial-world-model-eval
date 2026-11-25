"""动作敏感性与工况误差图：展示真实配对响应、方向/幅度/滞后及完整分组源数据。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import plot_week5 as shared

FAMILIES=[('persistence','Persistence','#777777'),('linear','Linear','#0072B2'),
          ('action_mlp','MLP','#D55E00'),('residual_mlp','Residual MLP','#009E73')]


def main():
    source=ROOT/'reports/week5/results/diagnostics'; shared.OUT=source/'figures'
    shared.OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,
        'axes.spines.top':False,'axes.spines.right':False})
    protocol=json.loads((source/'action/protocol.json').read_text())
    with np.load(source/'action/trajectories.npz') as data:
        plan=protocol['plan_names'].index('step_up'); reference=protocol['plan_names'].index('constant_0')
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        fig.subplots_adjust(left=.09,right=.98,bottom=.16,top=.88,hspace=.45,wspace=.28)
        fig.suptitle('Paired action response: 0 to 5 kW at action step 12',fontsize=14,y=.97)
        for i,weather in enumerate(('constant','daily')):
            scene=next(index for index,row in enumerate(protocol['scenes']) if row['initial']==[20.,19.] and row['weather']==weather)
            for j,variable in enumerate(('air','wall')):
                ax=axes[i,j]; steps=np.arange(81)
                truth=data['true_state'][scene,plan,:,j]-data['true_state'][scene,reference,:,j]
                ax.plot(steps,truth,color='black',lw=2.3,label='Simulator')
                for family,label,color in FAMILIES:
                    names=[family] if family in ('persistence','linear') else [f'{"mlp" if family=="action_mlp" else "residual"}_seed{s}' for s in (42,43,44)]
                    values=np.stack([data[name][scene,plan,:,j]-data[name][scene,reference,:,j] for name in names])
                    mean=values.mean(axis=0)
                    ax.plot(steps,mean,color=color,lw=1.5,ls='--',label=label)
                    if len(values)>1:
                        sd=values.std(axis=0,ddof=1); ax.fill_between(steps,mean-sd,mean+sd,color=color,alpha=.15)
                ax.axvline(12,color='#999999',lw=.8,ls=':')
                ax.set(title=f'{weather.capitalize()} outdoor / {variable}',xlabel='State step (10-min steps)',ylabel='Plan minus reference (°C)')
                ax.set_xticks([0,12,24,40,60,80]); ax.grid(alpha=.2)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=5,bbox_to_anchor=(.5,.045))
        fig.text(.5,.015,'Initial air/wall = 20/19 °C; same disturbance; no noise. Bands: training-seed SD, not prediction intervals.',ha='center',fontsize=8)
        shared.save(fig,'paired_action_response')
    summary=pd.read_csv(source/'action/response_summary.csv')
    plans=['constant_1','constant_2p5','constant_5','step_up','step_down','pulse','ramp']
    labels=['1 kW','2.5 kW','5 kW','Step up','Step down','Pulse','Ramp']
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.91,hspace=.65,wspace=.3)
    fig.suptitle('Paired response diagnostics across all 6 controlled scenes',fontsize=14,y=.97)
    fields=[('direction_accuracy_mean','Direction agreement (%)',100),
            ('abs_peak_magnitude_error_c_mean','Absolute peak-magnitude error (°C)',1),
            ('abs_lag_error_minutes_mean','Absolute 50%-response lag error (min)',1)]
    for i,(field,ylabel,factor) in enumerate(fields):
        for j,variable in enumerate(('air','wall')):
            ax=axes[i,j]
            for family,label,color in FAMILIES:
                group=summary[(summary.family==family)&(summary.variable==variable)].set_index('plan').reindex(plans)
                ax.plot(range(7),group[field].to_numpy()*factor,'o-',color=color,lw=1.5,ms=3,label=label)
            ax.set_xticks(range(7),labels,fontsize=8); ax.set(xlabel='Action plan',ylabel=ylabel,title=variable.capitalize())
            ax.grid(alpha=.2)
            if i==0: ax.set_ylim(-5,105)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.5,.045))
    fig.text(.5,.015,'Persistence never reaches the response threshold. Other missing lags are excluded; valid/requested counts are in the table.',ha='center',fontsize=8)
    shared.save(fig,'action_direction_amplitude_lag')
    summary.to_csv(shared.OUT/'action_plot_source.csv',index=False)

    # 热图展示所有预先固定 horizon 与分组，数字是回合等权后模型种子均值。
    groups=pd.read_csv(source/'decomposition/group_metrics_across_seeds.csv')
    groups=groups[(groups['mode']=='free_running')&groups.grouping.ne('all')]
    groups.to_csv(shared.OUT/'group_error_plot_source.csv',index=False)
    positive=groups.rmse_c_mean[np.isfinite(groups.rmse_c_mean)&(groups.rmse_c_mean>0)]
    low=min(.005,float(positive.min())/2); high=max(5.,float(positive.max()))
    use_log=(groups.rmse_c_mean.dropna()>0).all()
    norm=LogNorm(low,high) if use_log else Normalize(0,high)
    cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#eeeeee')
    group_definitions=[('power',['<1','[1,4)','>=4'],'Target-transition power (kW)'),
        ('policy',['mpc','random','rule'],'Collection policy'),
        ('initial_air',['<19','[19,21)','>=21'],'Episode initial air (°C)')]
    horizons=[1,5,10,20,40,80]
    for grouping,categories,label in group_definitions:
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        fig.subplots_adjust(left=.23,right=.98,bottom=.15,top=.88,hspace=.42,wspace=.55)
        fig.suptitle('Free-running terminal RMSE by '+label,fontsize=14,y=.97)
        row_index=[(family,category) for family,_,_ in FAMILIES for category in categories]
        row_labels=[f'{name} | {category}' for _,name,_ in FAMILIES for category in categories]
        for i,condition in enumerate(('recorded_observation','persistence_forecast')):
            for j,variable in enumerate(('air','wall')):
                group=groups[(groups.condition==condition)&(groups.variable==variable)&(groups.grouping==grouping)]
                matrix=group.pivot(index=['family','group'],columns='horizon',values='rmse_c_mean').reindex(index=pd.MultiIndex.from_tuples(row_index,names=['family','group']),columns=horizons)
                values=matrix.to_numpy(); ax=axes[i,j]
                ax.imshow(np.ma.masked_invalid(values),aspect='auto',norm=norm,cmap=cmap,interpolation='nearest')
                ax.set_xticks(range(6),horizons); ax.set_yticks(range(12),row_labels,fontsize=8)
                ax.set(xlabel='Evaluated horizon (10-min steps)',title=f'{"Recorded weather" if i==0 else "Outdoor held"} / {variable}')
                for row in range(12):
                    for col in range(6):
                        value=values[row,col]
                        ax.text(col,row,f'{value:.2g}' if np.isfinite(value) else 'NA',ha='center',va='center',fontsize=7,
                            color='white' if np.isfinite(value) and norm(value)<.65 else 'black')
        legend_values=[.01,.1,1.,high]
        fig.legend(handles=[Patch(facecolor=cmap(norm(value)),label=f'{value:g} °C') for value in legend_values],
            loc='lower center',ncol=4,bbox_to_anchor=(.5,.045))
        fig.text(.5,.015,'Cell text: RMSE in °C. Shared log colour scale; NA = no data. Group associations are not causal effects.',ha='center',fontsize=8)
        shared.save(fig,'group_error_'+grouping)


if __name__=='__main__': main()
