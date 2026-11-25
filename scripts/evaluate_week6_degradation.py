"""总说明：十一模型OOD/参考差值和同日期扰动变化曲线，保留配对身份与比较限制。"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluators.robustness import paired_metric_changes
from scripts import plot_week5 as shared


def main():
    base=ROOT/'reports/week6/results';out=base/'degradation';out.mkdir(exist_ok=True);shared.OUT=out
    refs=yaml.safe_load((ROOT/'configs/experiments/week6_boptest_extensions.yaml').read_text())['primary_reference_episodes']
    prediction=pd.read_csv(base/'residual_mlp/open_comparison.csv')
    control=pd.read_csv(base/'residual_mlp/control_comparison.csv');control=control[control.model.ne('embedded')]
    selected=prediction[prediction.episode_id.isin(refs)&prediction.condition.ne('ideal_weather_reference')]
    keys=['model','family','training_seed','episode_id','mode','horizon','variable','unit']
    paired=paired_metric_changes(selected,keys,'condition','persistence_weather',['rmse','mae','valid_fraction'])
    paired.to_csv(out/'paired_prediction_changes.csv',index=False)
    actual=control[control.episode_id.isin(refs)]
    actual=paired_metric_changes(actual,['model','family','training_seed','episode_id'],'condition','clean',
        ['cost','energy_kwh','comfort_degree_hours','fallback_count'])
    assert len(actual)==99
    actual.to_csv(out/'paired_control_changes.csv',index=False)
    # 天气日期无法与参考日期形成相同物理环境的干预对照，固定按模型配对均值。
    prediction['domain']=np.where(prediction.episode_id.isin(refs),'reference',np.where(prediction.episode_id.eq('id_test_day172'),'calendar_shift',prediction['split']))
    group=['model','family','training_seed','condition','mode','horizon','variable','unit']
    stats=prediction.groupby(group+['domain']).agg(rmse=('rmse','mean'),mae=('mae','mean'),dates=('episode_id','nunique'),valid_fraction=('valid_fraction','mean')).reset_index()
    weather=paired_metric_changes(stats,group,'domain','reference',['rmse','mae','valid_fraction'])
    weather.to_csv(out/'weather_model_matched_changes.csv',index=False)
    control_stats=control.groupby(['model','family','domain','condition']).agg(cost=('cost','mean'),energy_kwh=('energy_kwh','mean'),comfort_degree_hours=('comfort_degree_hours','mean'),dates=('episode_id','nunique')).reset_index()
    weather_control=paired_metric_changes(control_stats[control_stats.condition.eq('clean')],['model','family','condition'],'domain','reference',['cost','energy_kwh','comfort_degree_hours'])
    weather_control.to_csv(out/'weather_control_model_matched_changes.csv',index=False)
    colors={'persistence':'#888888','arx':'#0072B2','mlp':'#D55E00','gru':'#009E73','residual_mlp':'#CC79A7'}
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10,4));fig.subplots_adjust(left=.1,right=.98,bottom=.26,top=.8,wspace=.4)
    fig.suptitle('Within-date prediction change relative to clean observations',y=.97,fontsize=13)
    part=paired[paired['mode'].eq('free_running')&paired.horizon.eq(20)&paired.variable.eq('zone_c')]
    for ax,conditions,levels,title,xlabel in [(axes[0],['noise_0p1','noise_0p3'],[0,.1,.3],'Temperature noise','Noise standard deviation (°C)'),(axes[1],['missing_1','missing_4'],[0,1,4],'Supply sensor missing','Initial missing burst (15-min steps)')]:
        for family,color in colors.items():
            ep=part[part.family.eq(family)].groupby(['episode_id','condition']).rmse_delta.mean().reset_index()
            means=ep.groupby('condition').rmse_delta.mean().reindex(conditions)
            ax.plot(levels,[0,*means.to_list()],'o-',color=color,label=family,ms=4)
        ax.axhline(0,color='black',lw=.7,ls=':');ax.set(title=title,xlabel=xlabel,ylabel='20-step zone RMSE change (°C)');ax.set_xticks(levels);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,bbox_to_anchor=(.5,.04),fontsize=8)
    shared.save(fig,'paired_severity_changes')
    fig,axes=plt.subplots(1,2,figsize=(10,4));fig.subplots_adjust(left=.1,right=.98,bottom=.26,top=.8,wspace=.4)
    fig.suptitle('Weather holdout minus reference: model-matched date means',y=.97,fontsize=12)
    for ax,domain,title in zip(axes,['ood_cold','ood_warm'],['Cold-season holdout','Warm-weather holdout']):
        values=weather[weather.domain.eq(domain)&weather.condition.eq('persistence_weather')&weather['mode'].eq('free_running')&weather.horizon.eq(20)&weather.variable.eq('zone_c')]
        means=values.groupby('family').rmse_delta.mean().reindex(list(colors))
        ax.bar(range(5),means,color=list(colors.values()));ax.set_xticks(range(5),['Persist','ARX','MLP','GRU','Residual'],rotation=25)
        ax.axhline(0,color='black',lw=.8);ax.set(title=title,ylabel='20-step zone RMSE difference (°C)');ax.grid(axis='y',alpha=.2)
    fig.text(.5,.03,'Different dates, same checkpoint. Weather contrasts are not same-weather paired interventions.',ha='center',fontsize=8)
    shared.save(fig,'weather_minus_reference')
    (out/'README.md').write_text('''# OOD性能与扰动变化验收

差值均为“变化条件−参考条件”，误差/成本正值表示变差，负值表示改善。相对变化只在参考值绝对值大于1e-12时计算；参考接近零时保留NA。

paired_prediction_changes.csv包含同模型、同日期、同模式/horizon/变量的噪声或缺失相对无故障预测差值，保留每个日期与有效比例。paired_control_changes.csv包含99组同模型/同日期的目标+1°C、噪声和缺失相对clean的控制差值。目标+1°C属于任务OOD，评分标准同时变化，成本差不能直接当作预测模型性能损失。

weather_model_matched_changes.csv与weather_control_model_matched_changes.csv对相同检查点分别计算保留天气日期均值减三个参考日期均值。日期、初态及真实天气不同，只能称模型匹配的分布对比。不能把它写成同天气、同初态的严格配对因果干预。冷季保留温度仍在训练范围内；暖季208有明确支持外推。按日期重采样仍受小样本限制。

严重程度图覆盖开环噪声0/0.1/0.3°C与缺失0/1/4步。闭环目前只有clean、0.3°C、4步三个已执行端点，未伪造中间强度实验。曲线可以不单调，低强度观测故障不保证总误差必然上升。三个训练种子先在同日期平均，再平均日期，时间步不作为独立重复。
''',encoding='utf-8')
    result=dict(status='passed',models=11,reference_dates=3,paired_prediction_rows=len(paired),paired_control_rows=len(actual),
        weather_comparison='same checkpoint, different date groups',severity_noise=[0,.1,.3],severity_missing=[0,1,4])
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
