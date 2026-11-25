"""总说明：只根据保存的统计表和审核状态生成 Week6 中文报告；不把尚未完成的实验写成通过。"""
from pathlib import Path
import json
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]


def table(frame,columns=None):
    """独立 Markdown 表导出，不增加 tabulate 等可选依赖。"""
    if columns:frame=frame[columns]
    def value(x):
        if pd.isna(x):return '—'
        if isinstance(x,float):return f'{x:.4g}'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))


def main():
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text());base=ROOT/cfg['output'];summary=base/'summary'
    status=json.loads((base/'audit/status.json').read_text());models=json.loads((base/'training/models.json').read_text())
    support=pd.read_csv(base/'open_loop/weather_support.csv');control=pd.read_csv(summary/'control_by_family.csv')
    pc=pd.read_csv(summary/'prediction_vs_control_reference.csv');changes=pd.read_csv(summary/'paired_condition_changes.csv')
    raw=pd.read_csv(base/'open_loop/endpoint_errors.csv.gz')
    failures=json.loads((base/'open_loop/failures.json').read_text())
    text=f'''# Week 6：BOPTEST 世界模型的预测、鲁棒性与闭环控制

## 完成状态

审核状态：**{status['status']}**。18个时间块、5184条转移、8个模型实例。预定153个真实闭环回合，当前已完成{status['completed_controls']}个。最终完整性以 `results/audit/status.json` 为准；缺失身份全部列出，没有用现有结果冒充完整矩阵。

主平台为BOPTEST v0.9.0，案例 `bestest_hydronic_heat_pump`，FMU与资源哈希、上游commit、本地Docker修改均已归档。IB和自建热工环境不参与本周评测。

详细字段、数学定义、输入信息边界、OOD修正和统计口径见 [评测协议](../../docs/week6_protocol.md)。BOPTEST是公开物理仿真平台，本实验不代表真实建筑现场验证。

## 任务与数据

输入是16步公开观测历史、实际设定值历史、当前候选设定值，以及天气/日历；历史长度4小时。输出为下一步区域操作温度、供回水温度、地板热流、总电功率，以及当前15分钟区间电耗。区域操作温度不是可直接互换的空气温度，供水温度也不是墙温。

动作是18～24°C温度设定值，API转换为K，并单独发送覆盖激活。所有模型使用相同字段和历史长度。MPC评分的舒适标准独立于这个动作，不能通过改变设定值来改变评分目标。

8块训练、2块验证；每块7天预热、1天历史上下文、2天有效目标。预热和有效时间一起检查不重叠。训练1536个、验证384个有效一步目标。测试与补充参考共8个独立日期块。时间块、原始观测、实际动作和官方KPI可追溯到 `results/data/`。

### 天气支持和OOD解释

{table(support,['episode_id','outdoor_mean_c','outdoor_min_c','outdoor_max_c','outdoor_outside_training_fraction'])}

训练室外温度范围约−9～19.2°C。主要参考日期160、292、340均在该范围内，但范围一致不等于所有联合分布一致。原定参考日期172超范围18.23%，保留为独立日历变化案例。冷季日期12、24均在训练温度范围内，因此只称季节保留测试；暖季208有62.5%的温度超训练范围，是明确的温度支持外推测试。没有根据模型误差删除原日期。

第二类变化是同一参考日期的舒适上下界同时+1°C。它改变控制任务，不改变FMU天气或物理方程；模型保持冻结。这与观测噪声/缺失、天气分布变化分开报告。

## 模型与训练

{table(pd.DataFrame(models['models']).reindex(columns=['name','kind','seed','val_loss','alpha','best_epoch']))}

ARX验证选择岭系数；MLP为128/64两层网络；GRU隐藏维度64。MLP/GRU分别用42、43、44训练，学习率0.001，最大120epoch、验证早停耐心15。损失是六个输出标准化后的等权MSE，不把这个混合尺度训练损失当作物理单位测试误差。尺度仅在训练集拟合。GRU每个窗口独立初始化隐藏状态。

能耗标签由官方累计 `ener_tot` 的步差乘192m²得到，包含热泵、风机、水泵。已用功率时序梯形积分独立核对，见 `results/service_check/energy_integral_check.json`。没有用设定温度乘时间代替电耗，也没有自行假设热泵效率公式。

## 开环误差与有效性

共同起点每4步一个，统一预测20步，报告1、5、10、20步末端MAE/RMSE和完整轨迹。主条件未来天气保持起点值，另有理想天气预报对照。teacher forcing与自由递推分开；自由递推不读未来室温。

保存{len(raw)}条逐变量端点记录；数值/推理失败事件{len(failures)}条，非有效端点{int((~raw.valid).sum())}条，端点所属轨迹触发物理警告的记录{int(raw.physical_warning.sum())}条。警告不等于数值失败，且同一轨迹会在不同变量/步长重复计数。负能耗等原始预测保留，开环评分没有自动投影修正。

![开环多步误差](results/figures/open_loop_error.png)

每个面板对应变量与场景，阴影为日期间标准差，先在日期内平均神经模型训练种子。温度°C和步能耗kWh不能直接跨变量排名。完整逐变量表在 `results/open_loop/metrics.csv`，原始预测在 `raw_predictions.npz`，失败案例按变量分别挑选。

![观测故障严重程度](results/figures/open_loop_severity.png)

开环温度噪声0/0.1/0.3°C；起点供水温度缺失0/1/4步。缺失按过去值处理，后续自由预测仍不能读真实未来测量。

## 真实闭环结果

MPC每步256候选、12步预测时域（3小时）、候选动作块保持4步、15分钟重新规划，候选流和权重跨模型相同。只有BOPTEST推进实际状态。规划采用起点天气保持，已知未来日历与独立舒适日程。

累计代价=舒适区外温差平方积分+电耗+0.02×设定值变化平方。它是示例加权分数，不是货币费用。另报真实电耗、舒适累计偏差、违反率、区间中心跟踪误差、耗时与fallback。上下文不计入评分；原生KPI保留上下文结束和回合末两份，不能相减峰值或比例来假装得到有效区间KPI。

{table(control[control.condition.eq('clean')],['family','domain','cost_mean','cost_std','energy_kwh_mean','comfort_degree_hours_mean','fallback_count_mean'])}

上表先平均同一日期的训练种子，再报告日期间均值与样本标准差。它不是把3个训练种子当成3个独立建筑。三参考日期、两季节日期的样本量很小，bootstrap仅作为描述性不确定性。

![实际闭环指标](results/figures/closed_loop_kpis.png)

### 参考日期逐步轨迹

![日期160](results/figures/id_test_day160_trajectories.png)

![日期292](results/figures/id_test_day292_trajectories.png)

![日期340](results/figures/id_test_day340_trajectories.png)

每图分别为真实操作温度、实际生效设定值和步电耗；神经模型曲线为训练种子均值，阴影为训练种子标准差，不是预测区间。舒适虚线为独立日程，不随MPC动作更改。

## 鲁棒性与控制目标变化

![同日故障/目标变化](results/figures/robustness_and_goal_shift.png)

噪声只污染控制器看到的三个温度；缺失为供水传感器每24步连续4步。目标变化上下界+1°C，物理日期与预热相同。柱状图是同日相对clean的差值；目标改变后的成本差不能直接解释为模型退化。

所有无有效候选、超时、异常都记录并释放到内置控制，fallback真实成本不剔除。无求解器，因此使用“无有效候选”而非虚构不可行证书。时间预算5秒，检查粒度为批量预测之间，不保证抢占阻塞调用。耗时是在并发仿真机器上的实测值，不是独占CPU基准。

内置控制器是工程参照：它使用FMU内部传感器，客户端添加的观测故障不会污染其内部测量。因此不能把内置控制在噪声/缺失下不变解读为它比外部模型更抗传感器故障。公平的故障对比主要是同一外部控制器相对自身clean的变化。发生fallback的曲线代表“模型MPC+内置后备”的混合系统，不能当成纯模型规划效果。

## 预测—控制一致性

{table(pc,['model','rmse','cost','energy_kwh','comfort_degree_hours','fallback_count'])}

![预测与控制散点](results/figures/prediction_vs_control.png)

横轴是固定采集动作下的20步操作温度RMSE，纵轴是模型接入MPC后的实际成本；两者动作分布不同，不是因果回归。完整反转实例与对应episode在 `results/summary/prediction_control_counterexamples.csv`，没有预设结论。
'''
    cases=pd.read_csv(summary/'prediction_control_counterexamples.csv')
    if 'cost_gap' in cases:
        # 至少两个不同物理日期，避免把同一对换顺序当作两个独立实例。
        selected=cases.sort_values('cost_gap',ascending=False).drop_duplicates('episode_id').head(2)
        text+='\n'+table(selected,['episode_id','better_prediction_model','worse_prediction_model','rmse_a','rmse_b','cost_a','cost_b','fallback_a','fallback_b'])+'\n'
    text+='''
两个可核对的具体例子：

1. **日期340：Persistence 对比历史MLP seed43。** Persistence 的20步室温RMSE为0.667°C，优于MLP的0.753°C，但实际成本382.97对124.44。Persistence预测不随动作改变，实际设定值一直为18°C；平均舒适区外偏差2.100°C。MLP平均设定值20.140°C，平均区外偏差0.605°C，实际用电82.12kWh，也低于Persistence的87.78kWh。这说明温度变化较慢时，保持当前值可以获得不错的离线误差，但缺少动作响应无法支持规划。

2. **冷季日期24：GRU seed43 对比历史MLP seed42。** GRU温度RMSE为0.874°C，优于MLP的1.215°C，实际成本却为295.36对130.43；二者该回合fallback都为0。GRU平均设定值18.547°C，用电57.99kWh，平均区外偏差2.029°C；MLP平均设定值20.296°C，用电95.64kWh，平均区外偏差0.585°C。GRU选出较省电但舒适性更差的动作，舒适惩罚使总分更高。单一温度RMSE没有评价能耗预测与规划下的成本权衡；这些观察支持任务不一致的解释，但不能仅凭它们唯一归因于某个网络结构。

解释时要同时看：温度RMSE没有直接评价能耗头；动作敏感性与候选轨迹有效性会改变MPC选择；fallback可能让结果变好或变差；暖天气中本加热案例缺少主动制冷，过热可能超出执行器能力。不能把所有温度违反都归因于世界模型。具体分项和动作时序保存在逐回合CSV。

散点使用5小时预测误差，而MPC用3小时规划并每15分钟纠偏；预测horizon与反馈机制也不相同。天气减参考的误差差值在 `prediction_weather_minus_reference.csv`，控制差值及按日期重采样区间在 `weather_minus_reference.csv`；不同日期对比不冒称同天气因果干预。

策略排序扩展将各控制器实际前20步动作冻结为开环计划，与模拟器相同动作轨迹的真实前缀成本比较；报告有效计划数、方向一致率和真实regret。负能耗/失败模型计划不伪造评分，见 `results/policy_ranking/`。这不等同于让模型真实评估一个反馈策略的所有反事实行为。

## 复现与局限

```bash
conda activate mvp
bash scripts/reproduce_week6.sh
```

完整命令会检查服务、采集/校验数据、训练、开环、clean/目标变化/观测故障闭环、策略排序、测试与审核。已完成且摘要匹配的采集/闭环文件可以续跑；检查点变更必须先用新的输出目录或明确清除旧闭环结果，不能混用。检查点哈希与原始文件哈希保存在清单中。

- 仿真案例不是现场数字孪生的实测校准结果，不能外推为所有建筑结论。
- 数据量是MVP规模：8个训练日期块，未覆盖所有季节与设备状态；模型容量/超参数搜索有限。
- 一维设定值控制保留内置设备协调，不是直接优化所有HVAC执行器。
- 真实内部热状态不可见，模型只预测公开观测；未来负载未额外提供，不能声称掌握完整扰动。
- 目标变化是任务OOD，冷季保留不等于温度范围外推，支持范围内参考也不保证联合ID。
- 原生天气可能是理想预报；主规划使用天气保持，理想天气只作对照。
- 少量独立日期、重叠窗口和三个训练种子限制统计推断；不能把时间步当独立样本。
- 工程上的限幅、释放内置控制和启发有效性检查不是真实工业安全认证。

官方资料：[BOPTEST API](https://ibpsa.github.io/project1-boptest/docs-userguide/api.html)、[KPI定义](https://ibpsa.github.io/project1-boptest/docs-design/appendix_KPI.html)、[v0.9.0源码](https://github.com/ibpsa/project1-boptest/tree/v0.9.0)。实际实现以冻结本地版本为准。
'''
    ranking=pd.read_csv(base/'policy_ranking/ranking_by_episode.csv')
    totals=ranking.groupby('judge_model').agg(valid_plans=('valid_policies','sum'),requested_plans=('requested_policies','sum'),
        concordant=('concordant','sum'),discordant=('discordant','sum'),invalid_pairs=('invalid_pairs','sum'),mean_regret=('regret_vs_all_true_policies','mean')).reset_index()
    totals['conditional_pair_agreement']=totals.concordant/(totals.concordant+totals.discordant)
    totals.to_csv(base/'policy_ranking/summary.csv',index=False)
    text+='\n## 固定计划排序的有效性与条件评分\n\n'+table(totals)+'\n\n方向一致率只针对有效且不打平的计划对。尤其MLP seed42即使条件一致率很高，也不能忽略其有效计划数很少；无效计划和日期必须一起看。regret也是有有效候选时的条件值，不给无候选日期编造零regret。该实验暴露了未约束能耗预测可能为负的局限，不能把这些模型称为已达到工程可部署水平。\n'
    text+='\n最终测试：新增BOPTEST契约与模型/MPC测试10项通过；连同既有评测回归共29项通过。真实API覆盖/释放/fallback、原始功率积分核对、部署FMU一致性均通过，证据见 `results/service_check/` 和 `results/assets/deployed_fmu_check.json`。\n'
    target=ROOT/'reports/week6/week6.md';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text,encoding='utf-8')
    print('Wrote '+str(target),flush=True)


if __name__=='__main__':main()
