"""总说明：从逐日期原始统计生成 Week6 图，保留源表，统一 PNG/PDF/SVG 与面板对齐审计。"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import plot_week5 as shared

COLORS={'embedded':'#222222','persistence':'#888888','arx':'#0072B2','mlp':'#D55E00','gru':'#009E73'}
LABELS={'embedded':'Embedded','persistence':'Persistence','arx':'ARX','mlp':'History MLP','gru':'GRU'}


def main():
    if '--all-models' in sys.argv:
        from scripts.plot_week6_residual import main as residual_plot
        from scripts.plot_week6_open_updated import main as open_plot
        from scripts.plot_week6_closed_updated import main as closed_plot
        residual_plot()
        open_plot()
        closed_plot()
        return
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text());base=ROOT/cfg['output'];out=base/'figures';out.mkdir(parents=True,exist_ok=True);shared.OUT=out
    ext=yaml.safe_load((ROOT/'configs/experiments/week6_boptest_extensions.yaml').read_text());refs=ext['primary_reference_episodes']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    data=pd.read_csv(base/'open_loop/metrics_by_episode.csv');data['domain']=np.where(data.episode_id.isin(refs),'reference',np.where(data.episode_id.eq('id_test_day172'),'calendar_shift',data['split']))
    data.to_csv(out/'open_loop_source.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(12,7));fig.subplots_adjust(left=.08,right=.98,bottom=.17,top=.87,hspace=.48,wspace=.32)
    domains=[('reference','Reference dates'),('ood_cold','Cold-season holdout'),('ood_warm','Warm-weather holdout')]
    fig.suptitle('BOPTEST free-running error; outdoor forecast held at origin',y=.97,fontsize=13)
    for col,(domain,label) in enumerate(domains):
        for row,(variable,unit) in enumerate([('zone_c','°C'),('interval_energy','kWh')]):
            ax=axes[row,col];part=data[(data.domain==domain)&(data.condition=='persistence_weather')&(data['mode']=='free_running')&(data.variable==variable)]
            for family in ('persistence','arx','mlp','gru'):
                ep=part[part.family==family].groupby(['episode_id','horizon']).rmse.mean().reset_index()
                group=ep.groupby('horizon').rmse.agg(['mean','std']);x=group.index.to_numpy();mean=group['mean'].to_numpy();sd=group['std'].fillna(0).to_numpy()
                ax.plot(x,mean,'o-',color=COLORS[family],label=LABELS[family],ms=3)
                ax.fill_between(x,mean-sd,mean+sd,color=COLORS[family],alpha=.1)
            ax.set(title=label+' / '+('zone' if row==0 else 'step energy'),xlabel='Horizon (15-min steps)',ylabel=f'RMSE ({unit})');ax.set_xticks([1,5,10,20]);ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.5,.045))
    fig.text(.5,.015,'Seed means within date, then date mean ± SD. Three reference dates; two dates per seasonal holdout. Bands are not CIs.',ha='center',fontsize=8)
    shared.save(fig,'open_loop_error')
    control=pd.read_csv(base/'summary/control_by_episode.csv');control.to_csv(out/'control_source.csv',index=False)
    fig,axes=plt.subplots(3,3,figsize=(12,10));fig.subplots_adjust(left=.08,right=.98,bottom=.13,top=.9,hspace=.75,wspace=.35)
    fig.suptitle('BOPTEST real closed-loop outcomes: fixed MPC budget',fontsize=13,y=.97)
    families=['embedded','persistence','arx','mlp','gru']
    for row,(domain,label) in enumerate(domains):
        for col,(metric,unit) in enumerate([('cost','Weighted score'),('energy_kwh','Energy (kWh)'),('comfort_degree_hours','Comfort deviation (°C·h)')]):
            ax=axes[row,col];part=control[(control.domain==domain)&control.condition.eq('clean')]
            ep=part.groupby(['family','episode_id'])[metric].mean().reset_index();stats=ep.groupby('family')[metric].agg(['mean','std']).reindex(families)
            ax.bar(np.arange(5),stats['mean'],color=[COLORS[f] for f in families],width=.65)
            ax.errorbar(np.arange(5),stats['mean'],yerr=stats['std'].fillna(0),fmt='none',color='black',capsize=2,lw=.8)
            ax.set_xticks(np.arange(5),['Base','Persist','ARX','MLP','GRU'],fontsize=8);ax.set(title=label,ylabel=unit);ax.grid(axis='y',alpha=.2)
    fig.text(.5,.04,'Bars: training-seed means within date, then date means. Whiskers: date SD. All fallback trajectories stay in the score.',ha='center',fontsize=8)
    shared.save(fig,'closed_loop_kpis')
    pc=pd.read_csv(base/'summary/prediction_vs_control_by_episode.csv');pc=pc[pc.model.ne('embedded')]
    fig,axes=plt.subplots(1,3,figsize=(12,4));fig.subplots_adjust(left=.08,right=.98,bottom=.27,top=.78,wspace=.35)
    fig.suptitle('Offline prediction accuracy versus real closed-loop cost',y=.96,fontsize=13)
    for ax,(domain,label) in zip(axes,domains):
        part=pc[pc.domain==domain].groupby(['model','family'])[['rmse','cost']].mean().reset_index()
        for family in ('persistence','arx','mlp','gru'):
            g=part[part.family==family];ax.scatter(g.rmse,g.cost,label=LABELS[family],color=COLORS[family],s=35)
        ax.set(title=label,xlabel='20-step zone RMSE (°C)',ylabel='Closed-loop weighted score');ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.5,.07))
    fig.text(.5,.015,'Each point is one frozen model, averaged over paired dates. Offline actions and online feedback actions differ.',ha='center',fontsize=8)
    shared.save(fig,'prediction_vs_control')
    changes=pd.read_csv(base/'summary/paired_condition_changes.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,4));fig.subplots_adjust(left=.08,right=.98,bottom=.24,top=.78,wspace=.35)
    fig.suptitle('Same-date change relative to clean control',y=.96,fontsize=13)
    for ax,condition,label in zip(axes,['noise_0p3','missing_4','goal_plus1'],['Noise σ=0.3 °C','Supply sensor missing 4 steps','Comfort bounds +1 °C']):
        part=changes[changes.condition==condition]
        ep=part.groupby(['family','episode_id']).cost_delta.mean().reset_index();stats=ep.groupby('family').cost_delta.agg(['mean','std']).reindex(families)
        ax.bar(np.arange(5),stats['mean'],color=[COLORS[f] for f in families],width=.65)
        ax.errorbar(np.arange(5),stats['mean'],yerr=stats['std'].fillna(0),fmt='none',color='black',capsize=2,lw=.8)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(np.arange(5),['Base','Persist','ARX','MLP','GRU'],fontsize=8)
        ax.set(title=label,ylabel='Weighted score change');ax.grid(axis='y',alpha=.2)
    fig.text(.5,.04,'Within-date paired differences; whiskers show date SD. Goal shift changes the scoring task, not the plant dynamics.',ha='center',fontsize=8)
    shared.save(fig,'robustness_and_goal_shift')
    fig,axes=plt.subplots(1,2,figsize=(10,4));fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.78,wspace=.35)
    fig.suptitle('Open-loop observation robustness on reference dates',y=.96,fontsize=13)
    for ax,conditions,severity,title,xlabel in [(axes[0],['persistence_weather','noise_0p1','noise_0p3'],[0,.1,.3],'Temperature sensor noise','Noise standard deviation (°C)'),
        (axes[1],['persistence_weather','missing_1','missing_4'],[0,1,4],'Supply-water sensor missing','Initial missing burst (15-min steps)')]:
        selected=data[data.episode_id.isin(refs)&data['mode'].eq('free_running')&data.horizon.eq(20)&data.variable.eq('zone_c')]
        for family in ('persistence','arx','mlp','gru'):
            ep=selected[selected.family.eq(family)].groupby(['episode_id','condition']).rmse.mean().reset_index()
            stat=ep.groupby('condition').rmse.agg(['mean','std']).reindex(conditions)
            ax.plot(severity,stat['mean'],'o-',color=COLORS[family],label=LABELS[family],ms=4)
        ax.set(title=title,xlabel=xlabel,ylabel='20-step zone RMSE (°C)');ax.set_xticks(severity);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.5,.06))
    shared.save(fig,'open_loop_severity')
    # 三个预先固定参考日期分别画，均值/SD仅跨训练种子，内置控制不虚构种子重复。
    for eid in refs:
        fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True);fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.9,hspace=.42)
        fig.suptitle(eid+': actual BOPTEST closed-loop trajectories',fontsize=13,y=.97)
        for family in families:
            part=control[(control.episode_id==eid)&control.condition.eq('clean')&control.family.eq(family)]
            values=[]
            for identity in part.run_id:
                trace=pd.read_csv(base/'closed_loop'/(identity+'.csv'));trace=trace[~trace.context]
                values.append(trace[['zone_c','applied_action_c','energy_kwh']].to_numpy())
            if not values:continue
            array=np.stack(values);mean=array.mean(0);sd=array.std(0,ddof=1) if len(values)>1 else np.zeros_like(mean);hours=np.arange(1,len(mean)+1)*.25
            for row in range(3):
                axes[row].plot(hours,mean[:,row],color=COLORS[family],label=LABELS[family],lw=1.3)
                if len(values)>1:axes[row].fill_between(hours,mean[:,row]-sd[:,row],mean[:,row]+sd[:,row],color=COLORS[family],alpha=.1)
        if 'trace' in locals():
            axes[0].plot(hours,trace.lower_c,color='black',ls=':',lw=.8,label='Comfort bounds');axes[0].plot(hours,trace.upper_c,color='black',ls=':',lw=.8)
        for axis,label in zip(axes,['Zone operative temp. (°C)','Applied setpoint (°C)','Step electricity (kWh)']):axis.set_ylabel(label);axis.grid(alpha=.2)
        axes[-1].set_xlabel('Hours after evaluation start')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,.015),fontsize=8)
        shared.save(fig,eid+'_trajectories')


if __name__=='__main__':main()
