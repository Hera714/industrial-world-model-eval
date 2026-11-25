"""总说明：复用episode_bootstrap对完整日期块配对差值估计95%区间，窗口与训练种子不冒充独立episode。"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluators.robustness import episode_bootstrap


def summarize_differences(frame, group_columns, metrics, level):
    """每组每日期一个差值，缺失值保持不可用；同一家族先在同日期平均固定训练种子。"""
    identities=group_columns+['episode_id']
    if level=='model' and frame.duplicated(identities).any():raise ValueError('Repeated episode identity')
    if level=='family':
        def complete_mean(values):
            return values.mean() if values.notna().all() else np.nan
        frame=frame.groupby(identities,dropna=False)[metrics].agg(complete_mean).reset_index()
    records=[]
    for key,group in frame.groupby(group_columns,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        group=group.sort_values('episode_id')
        for metric in metrics:
            values=group[metric].to_numpy(float);finite=np.isfinite(values)
            row=dict(zip(group_columns,key));row.update(level=level,metric=metric,requested_episodes=len(values),valid_episodes=int(finite.sum()),
                episode_ids=json.dumps(group.episode_id.to_list()),seed=6042,bootstrap_repeats=2000,
                mean=float(values.mean()) if finite.all() else np.nan,lower95=np.nan,upper95=np.nan)
            if not finite.all():row['status']='unavailable_missing_episode_values'
            elif len(values)<2:row['status']='point_only_single_episode'
            else:
                ci=episode_bootstrap(values,seed=6042,repeats=2000)
                row.update(mean=ci['mean'],lower95=ci['lower95'],upper95=ci['upper95'],status='estimated')
            records.append(row)
    return pd.DataFrame(records)


def main():
    base=ROOT/'reports/week6/results';out=base/'bootstrap';out.mkdir(exist_ok=True)
    source=base/'degradation';control=pd.read_csv(source/'paired_control_changes.csv')
    prediction=pd.read_csv(source/'paired_prediction_changes.csv')
    all_control=pd.read_csv(base/'residual_mlp/control_comparison.csv')
    baseline=all_control[all_control.model.eq('embedded')][['episode_id','condition','cost','energy_kwh','comfort_degree_hours']]
    paired=all_control[all_control.model.ne('embedded')].merge(baseline,on=['episode_id','condition'],suffixes=('','_embedded'),validate='many_to_one',indicator=True)
    assert paired['_merge'].eq('both').all()
    for field in ['cost','energy_kwh','comfort_degree_hours']:paired[field+'_minus_embedded']=paired[field]-paired[field+'_embedded']
    paired.to_csv(out/'model_vs_embedded_by_episode.csv',index=False)
    outputs=[]
    for level in ('model','family'):
        who=['model','family'] if level=='model' else ['family']
        tasks=[('condition_control',control,who+['condition'],['cost_delta','energy_kwh_delta','comfort_degree_hours_delta','fallback_count_delta']),
            ('condition_prediction',prediction,who+['condition','mode','horizon','variable','unit'],['rmse_delta','mae_delta','valid_fraction_delta']),
            ('model_vs_embedded',paired,who+['domain','condition'],['cost_minus_embedded','energy_kwh_minus_embedded','comfort_degree_hours_minus_embedded'])]
        for name,frame,keys,metrics in tasks:
            summary=summarize_differences(frame,keys,metrics,level)
            path=out/(name+'_'+level+'_ci.csv');summary.to_csv(path,index=False);outputs.append(path)
    # 独立检查恒定差值和缺失处理，避免静默缩小样本集合。
    constant=episode_bootstrap([2.,2.,2.]);assert constant['lower95']==constant['upper95']==2.
    missing=pd.DataFrame(dict(model=['a','a'],episode_id=['x','y'],delta=[1.,np.nan]))
    assert summarize_differences(missing,['model'],['delta'],'model').iloc[0]['status']=='unavailable_missing_episode_values'
    (out/'README.md').write_text('''# Week6 episode bootstrap

估计量为完整日期块上的平均配对差值，使用固定seed6042、2000次有放回日期重采样、百分位95%区间。每次抽到某日期时，变化条件与参考条件保持同一配对。日期先排序，重复运行得到相同结果。没有重采样15分钟步或重叠窗口。

condition_control对应同模型/日期的goal_plus1、noise_0p3、missing_4相对clean。condition_prediction对应同模型/日期/模式/horizon/变量的观测故障相对无故障。model_vs_embedded对应同日期/条件的外部模型控制成本减内置控制。embedded具有内部传感器信息，客户端故障未施加到其内部，故障下比较只能视为工程参照。

model_ci保留11个实际模型实例。family_ci先在同日期平均该家族固定训练种子，再重采样日期，区间描述给定训练种子集合的日期不确定性，不估计重新训练的种子总体。训练种子不会被当成新的物理episode。

正的误差/成本差值表示变化条件更差，负值表示改善。目标变化同时改变评分任务，差值不能唯一归因于模型退化。有效比例的正差表示改善。区间包含0表示本小样本统计无法清晰区分差值方向，不等价于性能相同；多项比较未作多重检验修正。

同日期故障/目标测试只有3个参考日期，冷/暖季每类2个日期，单日历变化只有1个。单episode仅给点估计，区间置NA；任何缺失episode值使该差值区间不可用，保留请求数/有效数，不静默删除。小样本百分位区间可能不稳定，不能用2000次重复制造更多独立数据。

跨天气日期无法形成同初态/天气干预配对，因此不强行计算“配对天气OOD−ID区间”。已有天气日期组差值保留在../degradation；这里的配对区间严格按相同日期身份计算。
''',encoding='utf-8')
    inputs=[source/'paired_control_changes.csv',source/'paired_prediction_changes.csv',base/'residual_mlp/control_comparison.csv']
    status=dict(status='passed',models=11,seed=6042,repeats=2000,confidence=.95,method='paired episode percentile bootstrap',
        tables={p.name:len(pd.read_csv(p)) for p in outputs},
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        checks=['constant differences give degenerate interval','missing episode values do not get dropped'])
    (out/'verification.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))


if __name__=='__main__':main()
