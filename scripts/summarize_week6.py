"""总说明：汇总 BOPTEST 逐日期/训练种子结果、预测-控制表和 episode 配对差值，不删除 fallback 回合。"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluators.robustness import episode_bootstrap


def main():
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text());base=ROOT/cfg['output']
    extension=yaml.safe_load((ROOT/'configs/experiments/week6_boptest_extensions.yaml').read_text())
    ref=extension['primary_reference_episodes'];out=base/'summary';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in (base/'closed_loop').glob('*.json'):
        if path.name.endswith(('_forecast.json','_failure.json')):continue
        row=json.loads(path.read_text())
        if row.get('status')=='complete':records.append(row)
    if not records:raise ValueError('No completed controls')
    frame=pd.DataFrame(records)
    frame['domain']=np.where(frame.episode_id.isin(ref),'reference',np.where(frame.episode_id.eq('id_test_day172'),'calendar_shift',frame.split))
    frame.to_csv(out/'control_by_episode.csv',index=False)
    columns=['cost','energy_kwh','comfort_violation_rate','comfort_degree_hours','center_tracking_mae_c','decision_mean_ms','decision_p95_ms','fallback_count']
    permodel=frame.groupby(['model','family','training_seed','domain','condition'])[columns].agg(['mean','std','count'])
    permodel.columns=['_'.join(c) for c in permodel.columns];permodel.reset_index().to_csv(out/'control_by_model.csv',index=False)
    # 对神经模型先按同一日期平均训练种子，再比较日期；同时另存训练种子的真实分散程度。
    family_episode=frame.groupby(['family','episode_id','domain','condition'])[columns].mean().reset_index()
    family=family_episode.groupby(['family','domain','condition'])[columns].agg(['mean','std','count'])
    family.columns=['_'.join(c) for c in family.columns];family.reset_index().to_csv(out/'control_by_family.csv',index=False)
    baseline=frame[frame.model.eq('embedded')][['episode_id','condition','cost']].rename(columns={'cost':'embedded_cost'})
    paired=frame.merge(baseline,on=['episode_id','condition'],how='left',validate='many_to_one')
    paired['cost_delta_vs_embedded']=paired.cost-paired.embedded_cost;paired.to_csv(out/'paired_control_differences.csv',index=False)
    intervals=[]
    for (family,domain,condition),part in paired.groupby(['family','domain','condition']):
        byepisode=part.groupby('episode_id').cost_delta_vs_embedded.mean().dropna()
        if len(byepisode):intervals.append(dict(family=family,domain=domain,condition=condition,**episode_bootstrap(byepisode.to_numpy())))
    # 目标、噪声和缺失改变之前/之后都使用同一物理日期，配对差值按episode采样。
    clean=frame[frame.condition.eq('clean')][['model','episode_id']+columns]
    changes=frame[~frame.condition.eq('clean')].merge(clean,on=['model','episode_id'],suffixes=('','_clean'),validate='many_to_one')
    for field in columns:changes[field+'_delta']=changes[field]-changes[field+'_clean']
    changes.to_csv(out/'paired_condition_changes.csv',index=False)
    for (family,condition),part in changes.groupby(['family','condition']):
        delta=part.groupby('episode_id').cost_delta.mean().dropna()
        if len(delta):intervals.append(dict(family=family,domain='paired_condition_minus_clean',condition=condition,**episode_bootstrap(delta.to_numpy())))
    pd.DataFrame(intervals).to_csv(out/'episode_bootstrap.csv',index=False)
    # 天气时间块不是同一日期的干预配对：分别重采样日期，明确仅报告分布差异。
    contrasts=[];rng=np.random.default_rng(64042)
    for family_name,part in family_episode[family_episode.condition.eq('clean')].groupby('family'):
        reference=part[part.domain.eq('reference')]
        for domain in ('calendar_shift','ood_cold','ood_warm'):
            target=part[part.domain.eq(domain)]
            for metric in ('cost','energy_kwh','comfort_degree_hours'):
                a=reference[metric].to_numpy();b=target[metric].to_numpy()
                if len(a) and len(b):
                    draws=b[rng.integers(0,len(b),(2000,len(b)))].mean(1)-a[rng.integers(0,len(a),(2000,len(a)))].mean(1)
                    contrasts.append(dict(family=family_name,domain=domain,metric=metric,difference=float(b.mean()-a.mean()),
                        lower95=float(np.quantile(draws,.025)),upper95=float(np.quantile(draws,.975)),reference_dates=len(a),target_dates=len(b),
                        interpretation='Unpaired calendar-block distribution contrast; not a same-weather causal effect; tiny sample.'))
    pd.DataFrame(contrasts).to_csv(out/'weather_minus_reference.csv',index=False)
    pred=pd.read_csv(base/'open_loop/metrics_by_episode.csv')
    pred['domain']=np.where(pred.episode_id.isin(ref),'reference',np.where(pred.episode_id.eq('id_test_day172'),'calendar_shift',pred['split']))
    prediction_contrasts=[]
    perdate=pred.groupby(['family','episode_id','domain','condition','mode','horizon','variable','unit']).rmse.mean().reset_index()
    for key,part in perdate.groupby(['family','condition','mode','horizon','variable','unit']):
        a=part[part.domain.eq('reference')].rmse.to_numpy()
        for domain in ('calendar_shift','ood_cold','ood_warm'):
            b=part[part.domain.eq(domain)].rmse.to_numpy()
            if len(a) and len(b):
                prediction_contrasts.append(dict(zip(['family','condition','mode','horizon','variable','unit'],key))|
                    dict(domain=domain,rmse_difference=float(b.mean()-a.mean()),reference_dates=len(a),target_dates=len(b)))
    pd.DataFrame(prediction_contrasts).to_csv(out/'prediction_weather_minus_reference.csv',index=False)
    pred=pred[(pred.condition=='persistence_weather')&(pred['mode']=='free_running')&(pred.horizon==20)&(pred.variable=='zone_c')]
    pc=frame[frame.condition.eq('clean')].merge(pred[['model','episode_id','rmse','valid_fraction']],on=['model','episode_id'],how='left',validate='one_to_one')
    pc.to_csv(out/'prediction_vs_control_by_episode.csv',index=False)
    pc[pc.episode_id.isin(ref)].groupby(['model','family'])[['rmse','cost','energy_kwh','comfort_degree_hours','fallback_count']].mean().reset_index().to_csv(out/'prediction_vs_control_reference.csv',index=False)
    cases=[]
    for eid,part in pc.groupby('episode_id'):
        part=part[part.model.ne('embedded')]
        for a in part.itertuples():
            for b in part.itertuples():
                if a.rmse<b.rmse and a.cost>b.cost:
                    cases.append(dict(episode_id=eid,better_prediction_model=a.model,worse_prediction_model=b.model,
                        rmse_a=a.rmse,rmse_b=b.rmse,cost_a=a.cost,cost_b=b.cost,fallback_a=a.fallback_count,fallback_b=b.fallback_count,
                        cost_gap=a.cost-b.cost,interpretation='Association only: offline fixed-action error and online decision quality measure different tasks.'))
    pd.DataFrame(cases).sort_values('cost_gap',ascending=False).to_csv(out/'prediction_control_counterexamples.csv',index=False) if cases else (out/'prediction_control_counterexamples.csv').write_text('No ranking reversals observed\n')
    print(f'Summarized {len(frame)} complete controls; {len(cases)} ordered ranking reversals',flush=True)


if __name__=='__main__':main()
