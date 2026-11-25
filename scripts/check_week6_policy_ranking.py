"""总说明：核对固定计划的真实成本、共同初始历史/天气、排序和regret，并输出有效比例。"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluators.consistency import score_fixed_action_plans


def main():
    base=ROOT/'reports/week6/results';out=base/'residual_mlp/policy_ranking'
    plans=pd.read_csv(out/'plan_costs.csv',keep_default_na=False)
    plans['predicted_cost']=pd.to_numeric(plans.predicted_cost,errors='coerce')
    assert len(plans)==1056
    rows=[];histories={}
    for (eid,model),group in plans.groupby(['episode_id','judge_model']):
        assert len(group)==12
        for row in group.itertuples():
            trace=pd.read_csv(base/'closed_loop'/(eid+'__'+row.policy+'__clean.csv'))
            np.testing.assert_allclose(row.true_cost,trace.iloc[96:116].cost.sum(),atol=1e-9)
            initial=np.stack(trace.true_before.iloc[81:97].map(json.loads))
            actions=trace.applied_action_c.iloc[81:96].to_numpy()
            weather=np.stack(trace.weather.iloc[81:116].map(json.loads))
            if eid in histories:
                for actual,expected in zip((initial,actions,weather),histories[eid]):np.testing.assert_allclose(actual,expected,atol=1e-5)
            else:histories[eid]=(initial,actions,weather)
        score=score_fixed_action_plans(group.to_dict('records'))
        assert score['concordant']+score['discordant']+score['tied']+score['invalid_pairs']==66
        rows.append(dict(episode_id=eid,judge_model=model,**score))
    recalculated=pd.DataFrame(rows)
    original=pd.read_csv(out/'ranking_by_episode.csv').sort_values(['episode_id','judge_model']).reset_index(drop=True)
    recalculated=recalculated.sort_values(['episode_id','judge_model']).reset_index(drop=True)
    for key in ['concordant','discordant','tied','invalid_pairs','direction_agreement','valid_policies','regret_vs_all_true_policies']:
        np.testing.assert_allclose(recalculated[key],original[key],equal_nan=True,atol=1e-10)
    summary=[]
    for model,group in recalculated.groupby('judge_model'):
        c=int(group.concordant.sum());d=int(group.discordant.sum())
        summary.append(dict(model=model,requested_dates=len(group),dates_with_selection=int(group.selected_policy.notna().sum()),
            requested_plans=int(group.requested_policies.sum()),valid_plans=int(group.valid_policies.sum()),
            valid_plan_fraction=group.valid_policies.sum()/group.requested_policies.sum(),
            conditional_pair_agreement=c/(c+d) if c+d else np.nan,
            conditional_mean_regret=group.regret_vs_all_true_policies.mean(),
            tied_pairs=int(group.tied.sum()),invalid_pairs=int(group.invalid_pairs.sum())))
    pd.DataFrame(summary).to_csv(out/'acceptance_summary.csv',index=False)
    (out/'verification.json').write_text(json.dumps(dict(status='passed',judge_checkpoints=11,dates=8,plans_per_date=12,horizon_steps=20,
        model_plan_costs=1056,model_date_rankings=88,comparison_pairs_per_model_date=66,
        verified='Common initial history/weather; true prefix cost from simulator logs; rankings/regret recomputed',
        caveat='Fixed realized action plans, not counterfactual feedback-policy evaluation. Conditional agreement/regret must accompany validity.'),indent=2))
    (out/'acceptance_notes.md').write_text('''# 策略排序一致性验收

8个日期，各12条固定动作计划，来源于embedded与11个模型控制器实际执行的前20步（5小时）动作。每个世界模型对全部相同计划自由递推并预测成本；真实成本取相同初始历史下BOPTEST已执行轨迹的前20步。未来天气使用起点保持，已知舒适日程一致。

一致率=一致的非打平有效计划对数/(一致+反转)，成本差绝对值≤1e-6的任一侧打平单独统计。无效计划涉及的对数保留。regret=模型选中的计划真实成本−全部12计划中的最低真实成本；即使真实最佳计划被模型判无效，也留在比较基准。没有有效候选时regret记NA，不能记0。

acceptance_summary.csv并列有效计划比例、可选日期数、条件一致率与条件regret。高条件一致率与低有效比例可能同时出现，不能据此判模型最优。这是给定动作计划集内的选择后悔值，不是全局最优控制器的差距。原始预测前缀与失败原因均保留在raw_predictions.json和plan_costs.csv。
''',encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
