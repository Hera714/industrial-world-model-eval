"""独立统计单位汇总：先 episode、再模型种子；保留失败分母并输出配对差值。"""
from itertools import combinations
import numpy as np
import pandas as pd

CONTEXT = ['mode','condition','horizon','definition','variable']


def metric_columns(frame):
    """物理单位分开处理，不能把 IB 原生尺度错误命名成摄氏度。"""
    columns=[name for name in ('mae_c','rmse_c','mae_native','rmse_native') if name in frame]
    if not columns:
        raise ValueError('Missing episode error metrics')
    return columns


def summarize_episode_metrics(metrics, context=None):
    """输入必须已经是每回合一行的误差，绝不把重叠窗口当独立样本。"""
    context=list(CONTEXT if context is None else context)
    frame=metrics.copy()
    if 'unit' not in frame: frame['unit']='Celsius'
    keys=['model','family','seed']+context+['unit']
    errors=metric_columns(frame)
    if np.isinf(frame[errors].to_numpy()).any():
        raise ValueError('Episode errors must be finite or explicit NaN')
    if frame.empty or frame.duplicated(keys+['episode_id']).any():
        raise ValueError('Expected unique nonempty model/context/episode rows')
    counts=['requested','valid_origins','complete_origins','valid_points','failed_origins']
    if not np.isfinite(frame[counts].to_numpy()).all() or (frame[counts]<0).any().any():
        raise ValueError('Invalid episode counts')
    if (frame.requested<=0).any() or (frame.valid_origins>frame.requested).any() or (frame.failed_origins>frame.requested).any():
        raise ValueError('Counts exceed requested origins')
    # 缺少整行与一行 NaN 不是同一件事；前者必须报错，不能静默少算一个回合。
    for _, group in frame.groupby(context+['unit'],dropna=False,sort=False):
        expected=set(group.episode_id)
        for _, run in group.groupby('model',dropna=False):
            if set(run.episode_id)!=expected:
                raise ValueError('Missing episode rows: retain explicit failed rows instead of dropping episodes')
    aggregations={name+'_mean':(name,'mean') for name in errors}
    aggregations.update({name+'_episode_sd':(name,'std') for name in errors})
    aggregations.update(episodes=('episode_id','size'),episodes_with_valid_error=(errors[0],'count'))
    aggregations.update({name:(name,'sum') for name in counts})
    summary=frame.groupby(keys,dropna=False).agg(**aggregations).reset_index()
    summary['unavailable_origins']=summary.requested-summary.valid_origins
    summary['valid_fraction']=summary.valid_origins/summary.requested
    summary['failure_fraction']=summary.failed_origins/summary.requested
    seed_keys=['family']+context+['unit']
    aggregations={name+'_mean':(name+'_mean','mean') for name in errors}
    aggregations.update({name+'_seed_sd':(name+'_mean','std') for name in errors})
    aggregations.update(runs=('model','size'),runs_with_valid_error=(errors[0]+'_mean','count'),
        requested_per_run=('requested','first'),requested_model_origins=('requested','sum'),
        failed_model_origins=('failed_origins','sum'),unavailable_model_origins=('unavailable_origins','sum'),
        valid_fraction_mean=('valid_fraction','mean'),valid_fraction_seed_sd=('valid_fraction','std'),
        failure_fraction_mean=('failure_fraction','mean'),failure_fraction_seed_sd=('failure_fraction','std'))
    seeds=summary.groupby(seed_keys,dropna=False).agg(**aggregations).reset_index()
    return summary,seeds


def episode_seed_metrics(metrics):
    """在每个 episode 内汇总固定训练种子，用于配对；不增加独立 episode 数。"""
    frame=metrics.copy()
    if 'unit' not in frame: frame['unit']='Celsius'
    errors=metric_columns(frame)
    frame['_all_errors_finite']=np.isfinite(frame[errors].to_numpy()).all(axis=1)
    keys=['family']+CONTEXT+['unit','episode_id']
    expected=frame.groupby(['family','unit']).model.nunique().to_dict()
    agg={name+'_mean':(name,'mean') for name in errors}
    agg.update({name+'_seed_sd':(name,'std') for name in errors})
    agg.update(runs=('model','size'),valid_runs=('_all_errors_finite','sum'),requested=('requested','first'),
        min_valid_origins=('valid_origins','min'),min_complete_origins=('complete_origins','min'),
        max_failed_origins=('failed_origins','max'))
    result=frame.groupby(keys,dropna=False).agg(**agg).reset_index()
    result['expected_runs']=[expected[(row.family,row.unit)] for row in result.itertuples()]
    result['complete_support']=(result.runs==result.expected_runs)&(result.valid_runs==result.expected_runs)&\
        (result.min_complete_origins==result.requested)&(result.max_failed_origins==0)
    return result


def paired_episode_differences(episodes, information_groups=None):
    """同 episode 配对，A−B<0 表示 A 误差更小；缺失/失败配对保留但不伪造差值。"""
    groups=information_groups or {}
    errors=[c.removesuffix('_mean') for c in episodes if c.endswith('_mean') and c.startswith(('mae_','rmse_'))]
    keys=CONTEXT+['unit','episode_id']
    pairs=[]
    for family_a,family_b in combinations(sorted(episodes.family.unique()),2):
        a=episodes[episodes.family.eq(family_a)].drop(columns='family')
        b=episodes[episodes.family.eq(family_b)].drop(columns='family')
        # 单位或信息条件不同不做数值配对；相同单位下缺失的 episode 用外连接保留。
        context_a=set(map(tuple,a[CONTEXT+['unit']].drop_duplicates().to_numpy()))
        context_b=set(map(tuple,b[CONTEXT+['unit']].drop_duplicates().to_numpy()))
        if context_a!=context_b:
            raise ValueError('Paired families have different contexts or units')
        paired=a.merge(b,on=keys,how='outer',suffixes=('_a','_b'),indicator=True,validate='one_to_one')
        paired['family_a']=family_a; paired['family_b']=family_b
        paired['information_a']=groups.get(family_a,'unspecified')
        paired['information_b']=groups.get(family_b,'unspecified')
        paired['same_information_set']=paired.information_a.eq(paired.information_b)&paired.information_a.ne('unspecified')
        present=paired['_merge'].eq('both')
        valid=present&paired.complete_support_a.eq(True)&paired.complete_support_b.eq(True)&paired.requested_a.eq(paired.requested_b)
        paired['pair_status']=np.where(~present,'missing_episode',np.where(valid,'complete','incomplete_or_failed_support'))
        paired['paired_valid']=valid
        for name in errors:
            paired['delta_'+name]=(paired[name+'_mean_a']-paired[name+'_mean_b']).where(valid)
        paired['delta_max_run_failure_fraction']=paired.max_failed_origins_a/paired.requested_a-paired.max_failed_origins_b/paired.requested_b
        pairs.append(paired.drop(columns='_merge'))
    return pd.concat(pairs,ignore_index=True) if pairs else pd.DataFrame()


def summarize_pairs(pairs):
    """配对差先按回合计算，再报告回合均值和样本 SD；保留全部请求配对分母。"""
    if pairs.empty: return pd.DataFrame()
    keys=['family_a','family_b']+CONTEXT+['unit','information_a','information_b','same_information_set']
    delta=[name for name in pairs if name.startswith(('delta_mae_','delta_rmse_'))]
    agg=dict(requested_episode_pairs=('episode_id','size'),valid_episode_pairs=('paired_valid','sum'))
    agg.update({name+'_mean':(name,'mean') for name in delta})
    agg.update({name+'_episode_sd':(name,'std') for name in delta})
    result=pairs.groupby(keys,dropna=False).agg(**agg).reset_index()
    result['invalid_episode_pairs']=result.requested_episode_pairs-result.valid_episode_pairs
    result['paired_valid_fraction']=result.valid_episode_pairs/result.requested_episode_pairs
    return result
