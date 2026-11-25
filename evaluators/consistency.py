"""总说明：同日期模型预测误差与真实控制成本比较，保留输入/成本差异及可追溯的排序反转案例。"""
import pandas as pd


def compare_prediction_control(prediction, control, reference_episodes):
    """比较固定动作5小时室温预测与3小时滚动MPC，按日期配对，结果只表达关联。"""
    selected=prediction[prediction.condition.eq('persistence_weather')&prediction['mode'].eq('free_running')&prediction.horizon.eq(20)&prediction.variable.eq('zone_c')]
    selected=selected[['model','episode_id','rmse','valid_fraction']]
    actual=control[control.condition.eq('clean')&control.model.ne('embedded')]
    paired=actual.merge(selected,on=['model','episode_id'],how='left',validate='one_to_one')
    if paired.rmse.isna().any():raise ValueError('Missing paired prediction metric')
    summary=paired[paired.episode_id.isin(reference_episodes)].groupby(['model','family'])[['rmse','cost','energy_kwh','comfort_degree_hours','fallback_count']].mean().reset_index()
    cases=[]
    for eid,part in paired.groupby('episode_id'):
        for a in part.itertuples():
            for b in part.itertuples():
                if a.rmse<b.rmse and a.cost>b.cost:
                    cases.append(dict(episode_id=eid,model_a=a.model,model_b=b.model,rmse_a=a.rmse,rmse_b=b.rmse,
                        cost_a=a.cost,cost_b=b.cost,energy_a=a.energy_kwh,energy_b=b.energy_kwh,
                        comfort_a=a.comfort_degree_hours,comfort_b=b.comfort_degree_hours,
                        fallback_a=a.fallback_count,fallback_b=b.fallback_count,cost_gap=a.cost-b.cost,
                        trace_a='reports/week6/results/closed_loop/'+a.run_id+'.csv',
                        trace_b='reports/week6/results/closed_loop/'+b.run_id+'.csv'))
    return paired,summary,pd.DataFrame(cases).sort_values('cost_gap',ascending=False)


def score_fixed_action_plans(plans, tie_tolerance=1e-6):
    """固定动作计划的条件排序一致率与真实regret，保留无效候选、打平和无可选计划。"""
    import itertools
    import numpy as np
    if len({row['policy'] for row in plans}) != len(plans):
        raise ValueError('Duplicate fixed action plan')
    if not all(np.isfinite(row['true_cost']) for row in plans):
        raise ValueError('Nonfinite simulator cost')
    valid = [row for row in plans if row['valid'] and np.isfinite(row['predicted_cost'])]
    valid_names = {row['policy'] for row in valid}
    concordant = discordant = tied = invalid = 0
    for a, b in itertools.combinations(plans, 2):
        if a['policy'] not in valid_names or b['policy'] not in valid_names:
            invalid += 1
            continue
        actual = a['true_cost'] - b['true_cost']
        predicted = a['predicted_cost'] - b['predicted_cost']
        if abs(actual) <= tie_tolerance or abs(predicted) <= tie_tolerance:
            tied += 1
        elif np.sign(actual) == np.sign(predicted):
            concordant += 1
        else:
            discordant += 1
    selected = min(valid, key=lambda row: row['predicted_cost']) if valid else None
    return dict(concordant=concordant, discordant=discordant, tied=tied, invalid_pairs=invalid,
        direction_agreement=concordant / (concordant + discordant) if concordant + discordant else np.nan,
        valid_policies=len(valid), requested_policies=len(plans),
        selected_policy=selected['policy'] if selected else None,
        selected_true_cost=selected['true_cost'] if selected else np.nan,
        regret_vs_all_true_policies=selected['true_cost']-min(row['true_cost'] for row in plans) if selected else np.nan)
