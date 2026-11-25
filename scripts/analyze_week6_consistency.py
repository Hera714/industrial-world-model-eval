"""总说明：十一模型预测-控制表、散点与不同日期的可追溯反例，逐步分项来自真实轨迹。"""
from pathlib import Path
import sys,json
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluators.consistency import compare_prediction_control
from scripts import plot_week5 as shared


def main():
    base=ROOT/'reports/week6/results';out=base/'consistency';out.mkdir(exist_ok=True);shared.OUT=out
    refs=yaml.safe_load((ROOT/'configs/experiments/week6_boptest_extensions.yaml').read_text())['primary_reference_episodes']
    prediction=pd.read_csv(base/'residual_mlp/open_comparison.csv');control=pd.read_csv(base/'residual_mlp/control_comparison.csv')
    paired,summary,cases=compare_prediction_control(prediction,control,refs)
    assert len(paired)==88 and summary.model.nunique()==11
    paired.to_csv(out/'prediction_vs_control_by_episode.csv',index=False);summary.to_csv(out/'prediction_vs_control_reference.csv',index=False)
    cases.to_csv(out/'all_ranking_reversals.csv',index=False)
    first=cases[cases.model_a.eq('persistence')].head(1)
    second=cases[cases.model_a.ne('persistence')&cases.model_b.ne('persistence')&~cases.episode_id.isin(first.episode_id)&cases.fallback_a.eq(0)&cases.fallback_b.eq(0)].head(1)
    chosen=pd.concat([first,second]);assert len(chosen)==2
    chosen.to_csv(out/'selected_cases.csv',index=False)
    lines=['# 任务7：预测—控制一致性验收','',
        '全部11模型在共同8日期上配对。主散点取三个参考日期均值；横轴为固定动作、天气保持条件下20步（5小时）区域温度RMSE，纵轴为真实BOPTEST闭环累计成本。MPC时域为12步（3小时）并每15分钟纠偏，预测/控制的动作分布及目标均不同，散点表达关联。','']
    for n,row in enumerate(chosen.itertuples(),1):
        lines += [f'## 案例{n} · {row.episode_id}', '',
            f'{row.model_a} 温度RMSE {row.rmse_a:.3f}°C，小于 {row.model_b} 的 {row.rmse_b:.3f}°C；闭环成本为 {row.cost_a:.2f} 与 {row.cost_b:.2f}。',
            f'实际能耗 {row.energy_a:.2f}/{row.energy_b:.2f} kWh，舒适累计偏差 {row.comfort_a:.2f}/{row.comfort_b:.2f} °C·h，fallback {row.fallback_a}/{row.fallback_b} 次。',
            'Persistence缺乏动作响应，离线温度变化慢时可以获得较小误差，规划仍可能选择不利设定值。' if row.model_a=='persistence' else '两个模型的实际能耗与舒适偏差体现不同的动作权衡。单一温度RMSE没有评价能耗头、候选动作响应或反馈纠偏，不能从排序反转唯一归因于网络结构。',
            f'轨迹A：`{row.trace_a}`；轨迹B：`{row.trace_b}`。','']
    lines+=['![共同参考日期散点](prediction_vs_control.png)','','fallback曲线属于模型MPC与内置后备的混合系统。原始状态与动作、舒适界限和成本均保留于逐步CSV；无效开环预测应连同有效比例查看。']
    (out/'cases.md').write_text('\n'.join(lines),encoding='utf-8')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8,5));fig.subplots_adjust(left=.13,right=.97,bottom=.23,top=.88)
    colors={'persistence':'#888888','arx':'#0072B2','mlp':'#D55E00','gru':'#009E73','residual_mlp':'#CC79A7'}
    for family,color in colors.items():
        part=summary[summary.family.eq(family)];ax.scatter(part.rmse,part.cost,color=color,label=family,s=45)
    ax.set(xlabel='20-step zone RMSE (°C)',ylabel='Actual closed-loop weighted cost',title='Prediction versus control: 11 checkpoints, three paired dates');ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=5,bbox_to_anchor=(.5,.04),fontsize=8)
    shared.save(fig,'prediction_vs_control')
    (out/'verification.json').write_text(json.dumps(dict(status='passed',models=11,paired_model_dates=88,reference_dates=3,
        ranking_reversal_cases=len(cases),selected_cases=2,distinct_case_dates=chosen.episode_id.nunique(),
        caveat='Offline fixed-action prediction and closed-loop decision outcomes are different estimands; no causal claim.'),indent=2))
    print(chosen[['episode_id','model_a','model_b','rmse_a','rmse_b','cost_a','cost_b']].to_string(index=False))


if __name__=='__main__':main()
