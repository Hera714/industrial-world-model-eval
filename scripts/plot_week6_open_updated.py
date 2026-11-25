"""总说明：将残差MLP加入任务4的多步误差与观测故障图。"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import plot_week5 as shared


def main():
    base=ROOT/'reports/week6/results/residual_mlp';shared.OUT=base/'figures'
    data=pd.read_csv(base/'open_comparison.csv');refs=['id_test_day160','id_test_day292','id_test_day340']
    data=data[data['mode'].eq('free_running')]
    families={'persistence':('Persistence','#888888'),'arx':('ARX','#0072B2'),'mlp':('History MLP','#D55E00'),'gru':('GRU','#009E73'),'residual_mlp':('Residual MLP','#CC79A7')}
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(12,7));fig.subplots_adjust(left=.08,right=.98,bottom=.16,top=.87,hspace=.5,wspace=.35)
    fig.suptitle('BOPTEST free-running error, including residual MLP',y=.97,fontsize=13)
    for col,(split,title) in enumerate([('reference','Reference dates'),('ood_cold','Cold-season holdout'),('ood_warm','Warm-weather holdout')]):
        selected=data[data.episode_id.isin(refs)] if split=='reference' else data[data['split'].eq(split)]
        for row,(variable,label) in enumerate([('zone_c','Zone RMSE (°C)'),('interval_energy','Step energy RMSE (kWh)')]):
            ax=axes[row,col]
            for family,(name,color) in families.items():
                part=selected[selected.family.eq(family)&selected.variable.eq(variable)&selected.condition.eq('persistence_weather')]
                stats=part.groupby(['episode_id','horizon']).rmse.mean().groupby('horizon').agg(['mean','std'])
                x=stats.index.to_numpy();m=stats['mean'].to_numpy();s=stats['std'].fillna(0).to_numpy()
                ax.plot(x,m,'o-',color=color,label=name,ms=3);ax.fill_between(x,m-s,m+s,color=color,alpha=.08)
            ax.set(title=title,xlabel='Horizon (15-min steps)',ylabel=label);ax.set_xticks([1,5,10,20]);ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=5,bbox_to_anchor=(.5,.04))
    fig.text(.5,.01,'Weather held at origin. Seed means within date, then date mean ± SD; bands are not confidence intervals.',ha='center',fontsize=8)
    shared.save(fig,'open_loop_error_with_residual')
    fig,axes=plt.subplots(1,2,figsize=(10,4));fig.subplots_adjust(left=.09,right=.98,bottom=.26,top=.8,wspace=.35)
    fig.suptitle('Open-loop observation robustness, including residual MLP',y=.97,fontsize=13)
    selected=data[data.episode_id.isin(refs)&data.variable.eq('zone_c')&data.horizon.eq(20)]
    for ax,conditions,levels,title,xlabel in [(axes[0],['persistence_weather','noise_0p1','noise_0p3'],[0,.1,.3],'Temperature sensor noise','Noise standard deviation (°C)'),(axes[1],['persistence_weather','missing_1','missing_4'],[0,1,4],'Supply-water sensor missing','Initial missing burst (15-min steps)')]:
        for family,(name,color) in families.items():
            group=selected[selected.family.eq(family)].groupby(['episode_id','condition']).rmse.mean().groupby('condition').mean().reindex(conditions)
            ax.plot(levels,group,'o-',label=name,color=color,ms=4)
        ax.set(title=title,xlabel=xlabel,ylabel='20-step zone RMSE (°C)');ax.set_xticks(levels);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,bbox_to_anchor=(.5,.04),fontsize=8)
    shared.save(fig,'open_loop_severity_with_residual')


if __name__=='__main__':main()
