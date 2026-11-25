"""总说明：五类模型统一比较图，复用相同日期、信息集、horizon和真实闭环评分。"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import plot_week5 as shared


def main():
    base=ROOT/'reports/week6/results/residual_mlp';shared.OUT=base/'figures';shared.OUT.mkdir(exist_ok=True)
    frame=pd.read_csv(base/'open_comparison.csv');control=pd.read_csv(base/'control_comparison.csv')
    refs=['id_test_day160','id_test_day292','id_test_day340']
    colors={'persistence':'#888888','arx':'#0072B2','mlp':'#D55E00','gru':'#009E73','residual_mlp':'#CC79A7'}
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(12,4));fig.subplots_adjust(left=.08,right=.98,bottom=.26,top=.78,wspace=.4)
    fig.suptitle('BOPTEST residual history MLP: same-input comparison',y=.97,fontsize=13)
    for family,color in colors.items():
        part=frame[frame.episode_id.isin(refs)&frame.family.eq(family)&frame.condition.eq('persistence_weather')&frame['mode'].eq('free_running')]
        for ax,variable,label in zip(axes[:2],['zone_c','interval_energy'],['Zone RMSE (°C)','Step energy RMSE (kWh)']):
            group=part[part.variable.eq(variable)].groupby(['episode_id','horizon']).rmse.mean().groupby('horizon').mean()
            ax.plot(group.index,group,'o-',color=color,label=family,ms=4);ax.set(xlabel='Horizon (15-min steps)',ylabel=label);ax.set_xticks([1,5,10,20]);ax.grid(alpha=.2)
    families=list(colors);stats=control[control.episode_id.isin(refs)&control.condition.eq('clean')].groupby(['family','episode_id']).cost.mean().groupby('family').agg(['mean','std']).reindex(families)
    axes[2].bar(range(5),stats['mean'],color=[colors[f] for f in families]);axes[2].errorbar(range(5),stats['mean'],yerr=stats['std'],fmt='none',color='black',capsize=2)
    axes[2].set_xticks(range(5),['Persist','ARX','MLP','GRU','Residual'],rotation=25);axes[2].set_ylabel('Actual MPC weighted cost');axes[2].grid(axis='y',alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,bbox_to_anchor=(.5,.04),fontsize=8)
    shared.save(fig,'residual_comparison')


if __name__=='__main__':main()
