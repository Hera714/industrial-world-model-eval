"""总说明：策略排序有效性与episode配对成本置信区间的可追溯图。"""
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
    base=ROOT/'reports/week6/results';shared.OUT=base/'p1_figures';shared.OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    rank=pd.read_csv(base/'residual_mlp/policy_ranking/acceptance_summary.csv')
    rank.columns=rank.columns.str.strip()
    rank['model']=rank['model'].str.strip()
    names=rank.model.str.replace('residual_mlp','Residual',regex=False).str.replace('_seed',' ',regex=False)
    y=np.arange(len(rank))
    fig,axes=plt.subplots(1,3,figsize=(12,6));fig.subplots_adjust(left=.15,right=.98,bottom=.16,top=.83,wspace=.38)
    fig.suptitle('Fixed-plan ranking: agreement, validity and regret',y=.96,fontsize=13)
    for ax,column,title,scale in [(axes[0],'conditional_pair_agreement','Agreement on valid pairs (%)',100),
        (axes[1],'valid_plan_fraction','Valid plan evaluations (%)',100),(axes[2],'conditional_mean_regret','Mean regret when selection exists',1)]:
        ax.barh(y,rank[column]*scale,color='#0072B2');ax.set_yticks(y,names if ax is axes[0] else []);ax.invert_yaxis();ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2)
        if scale==100:ax.set_xlim(0,105)
    fig.text(.5,.04,'11 judges; 8 dates × 12 plans. Agreement/regret are conditional; validity must be read alongside them.',ha='center',fontsize=8)
    shared.save(fig,'policy_ranking_summary')
    ci=pd.read_csv(base/'bootstrap/condition_control_model_ci.csv');models=rank.model.to_list()
    ci.columns=ci.columns.str.strip()
    for column in ['model','condition','metric']:ci[column]=ci[column].str.strip()
    fig,axes=plt.subplots(1,3,figsize=(12,6));fig.subplots_adjust(left=.15,right=.98,bottom=.17,top=.83,wspace=.4)
    fig.suptitle('Paired episode bootstrap: change in actual closed-loop cost',y=.96,fontsize=13)
    for ax,condition,title in zip(axes,['noise_0p3','missing_4','goal_plus1'],['Noise σ=0.3 °C','Supply sensor missing 4 steps','Comfort bounds +1 °C']):
        group=ci[ci.condition.eq(condition)&ci.metric.eq('cost_delta')].set_index('model').reindex(models)
        for i,row in enumerate(group.itertuples()):
            ax.plot([row.lower95,row.upper95],[i,i],color='#0072B2',lw=1.5)
            ax.plot(row.mean,i,'o',color='#D55E00',ms=4)
        ax.axvline(0,color='black',ls=':',lw=1);ax.set_yticks(y,names if ax is axes[0] else []);ax.invert_yaxis();ax.set(title=title,xlabel='Changed minus clean weighted cost');ax.grid(axis='x',alpha=.2)
    fig.text(.5,.055,'Dots: paired-date mean; lines: percentile 95% CI, 2,000 date resamples. Three dates; seed fixed at 6042.',ha='center',fontsize=8)
    fig.text(.5,.025,'Goal shift changes the scoring task. Intervals describe date uncertainty for each fixed trained checkpoint.',ha='center',fontsize=8)
    shared.save(fig,'episode_bootstrap_cost_changes')


if __name__=='__main__':main()
