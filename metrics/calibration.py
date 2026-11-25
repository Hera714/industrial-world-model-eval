"""Ensemble 区间校准：仅按验证集拟合逐 horizon/变量系数，保留失败预测的分母。"""
import math
import numpy as np


def ensemble_moments(members, scale_floor_c):
    """每个成员先独立递推再聚合；任一成员缺失则该位置不产生有效区间。"""
    values=np.asarray(members,dtype=float)
    if values.ndim!=4 or values.shape[0]<2 or scale_floor_c<=0 or not np.isfinite(scale_floor_c):
        raise ValueError('Expected (members,origins,horizon,variables) and positive scale floor')
    valid=np.isfinite(values).all(axis=0)
    safe=np.where(np.isfinite(values),values,0.)
    mean=safe.mean(axis=0); sd=safe.std(axis=0,ddof=1)
    mean=np.where(valid,mean,np.nan); sd=np.where(valid,sd,np.nan)
    return mean,sd,np.maximum(sd,scale_floor_c)


def fit_interval_scale(members, actual, nominal=.9, scale_floor_c=.01, split='val'):
    """验证集绝对误差/ensemble 尺度分数，取 ceil((n+1)*nominal) 阶统计量。

    重叠窗口不满足常规可交换假设，此处系数用于经验覆盖评估，不承诺严格覆盖。
    样本不足时返回 NaN 和明确状态，不夹成最大分数冒充可用的名义区间。
    """
    if split!='val': raise ValueError('Calibration fitting is restricted to validation data')
    if not 0<nominal<1: raise ValueError('Nominal coverage must lie in (0,1)')
    mean,sd,scale=ensemble_moments(members,scale_floor_c)
    actual=np.asarray(actual,dtype=float)
    if actual.shape!=mean.shape or not np.isfinite(actual).all(): raise ValueError('Invalid calibration targets')
    scores=np.abs(actual-mean)/scale
    q=np.full(mean.shape[1:],np.nan); counts=np.zeros_like(q,dtype=int)
    ranks=np.zeros_like(q,dtype=int); status=np.full(q.shape,'insufficient_valid_scores',dtype=object)
    for h in range(q.shape[0]):
        for variable in range(q.shape[1]):
            values=scores[:,h,variable]; values=values[np.isfinite(values)]
            n=len(values); rank=math.ceil((n+1)*nominal)
            counts[h,variable]=n; ranks[h,variable]=rank
            if n and rank<=n:
                q[h,variable]=np.partition(values,rank-1)[rank-1]; status[h,variable]='fitted'
    return dict(q=q,valid_scores=counts,rank=ranks,status=status,requested=mean.shape[0])


def interval_arrays(members,q,scale_floor_c):
    """测试集只应用已冻结系数，绝不从测试目标计算参数。"""
    mean,sd,scale=ensemble_moments(members,scale_floor_c)
    q=np.asarray(q,dtype=float)
    if q.shape!=mean.shape[1:] or (q[np.isfinite(q)]<0).any(): raise ValueError('Invalid calibrated scale shape/value')
    return dict(mean=mean,sd=sd,scale=scale,lower=mean-q*scale,upper=mean+q*scale)


def interval_metrics(actual,lower,upper):
    """覆盖率以所有请求点为主分母；条件覆盖率/宽度同时保留有效点计数。"""
    actual,lower,upper=map(lambda x:np.asarray(x,dtype=float),(actual,lower,upper))
    if actual.shape!=lower.shape or actual.shape!=upper.shape or not actual.size or not np.isfinite(actual).all():
        raise ValueError('Invalid interval evaluation arrays')
    valid=np.isfinite(lower)&np.isfinite(upper)&(lower<=upper)
    covered=valid&(actual>=lower)&(actual<=upper)
    width=np.where(valid,upper-lower,np.nan)
    n=int(valid.sum()); requested=actual.size
    return dict(requested=requested,valid=n,invalid=requested-n,covered=int(covered.sum()),
        valid_fraction=n/requested,coverage_all=float(covered.sum()/requested),
        coverage_valid=float(covered.sum()/n) if n else np.nan,
        mean_width_c=float(width[valid].mean()) if n else np.nan)
