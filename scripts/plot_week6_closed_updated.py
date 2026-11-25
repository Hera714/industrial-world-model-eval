"""总说明：任务6配对闭环图，包含残差MLP，神经模型先按同日期平均训练种子。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import plot_week5 as shared


def main():
    base=ROOT/'reports/week6/results';output=base/'residual_mlp/figures';shared.OUT=output
    data=pd.read_csv(base/'residual_mlp/control_comparison.csv')
    data=data[data.condition.eq('clean')]
    refs=['id_test_day160','id_test_day292','id_test_day340']
    families={'embedded':('Embedded','#222222'),'persistence':('Persistence','#888888'),'arx':('ARX','#0072B2'),
        'mlp':('History MLP','#D55E00'),'gru':('GRU','#009E73'),'residual_mlp':('Residual MLP','#CC79A7')}
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(12,4));fig.subplots_adjust(left=.08,right=.98,bottom=.26,top=.8,wspace=.4)
    fig.suptitle('Paired BOPTEST closed-loop outcomes on three reference dates',y=.97,fontsize=13)
    selected=data[data.episode_id.isin(refs)]
    for ax,metric,label in zip(axes,['cost','energy_kwh','comfort_degree_hours'],['Weighted cost','Electricity (kWh)','Comfort deviation (°C·h)']):
        stats=selected.groupby(['family','episode_id'])[metric].mean().groupby('family').agg(['mean','std']).reindex(list(families))
        ax.bar(range(6),stats['mean'],color=[value[1] for value in families.values()])
        ax.errorbar(range(6),stats['mean'],yerr=stats['std'],fmt='none',color='black',capsize=2)
        ax.set_xticks(range(6),['Base','Persist','ARX','MLP','GRU','Residual'],rotation=30);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.text(.5,.02,'Seed means within date, then date mean. Whiskers: date SD. All fallback costs included.',ha='center',fontsize=8)
    shared.save(fig,'closed_loop_kpis_with_residual')
    for eid in refs:
        fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True);fig.subplots_adjust(left=.1,right=.98,bottom=.16,top=.9,hspace=.4)
        fig.suptitle(eid+' paired actual closed-loop trajectories',y=.97,fontsize=13)
        for family,(name,color) in families.items():
            runs=data[data.episode_id.eq(eid)&data.family.eq(family)]
            values=[]
            for identity in runs.run_id:
                trace=pd.read_csv(base/'closed_loop'/(identity+'.csv'));trace=trace[~trace.context]
                values.append(trace[['zone_c','applied_action_c','energy_kwh']].to_numpy())
            array=np.stack(values);mean=array.mean(0);sd=array.std(0,ddof=1) if len(array)>1 else np.zeros_like(mean)
            hours=np.arange(1,len(mean)+1)*.25
            for j,ax in enumerate(axes):
                ax.plot(hours,mean[:,j],color=color,label=name,lw=1.3)
                if len(array)>1:ax.fill_between(hours,mean[:,j]-sd[:,j],mean[:,j]+sd[:,j],color=color,alpha=.09)
        axes[0].plot(hours,trace.lower_c,'k:',lw=.8,label='Comfort bounds');axes[0].plot(hours,trace.upper_c,'k:',lw=.8)
        for ax,label in zip(axes,['Zone operative temp. (°C)','Applied setpoint (°C)','Step electricity (kWh)']):ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[-1].set_xlabel('Hours after evaluation start')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.5,.025),fontsize=8)
        shared.save(fig,eid+'_closed_with_residual')


if __name__=='__main__':main()
