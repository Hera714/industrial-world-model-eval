"""经验概率评分：等权离散 ensemble 的精确 CRPS；不额外假设正态密度或计算 NLL。"""
import numpy as np


def empirical_crps_components(actual,members):
    """成员轴在第0维，返回逐样本 CRPS 及两项分解，单位与目标相同。

    CRPS = mean_m |x_m-y| - sum_mn |x_m-x_n|/(2M²)。
    分母是 M²，非 fair/IID 修正的 M(M−1)。排序恒等式避免构造 M×M 数组。
    任一成员非有限时整份固定成员分布记为不可用，不重新归一化剩余成员。
    """
    actual=np.asarray(actual,dtype=float); members=np.asarray(members,dtype=float)
    if not actual.size or members.ndim!=actual.ndim+1 or members.shape[1:]!=actual.shape or members.shape[0]<1:
        raise ValueError('Expected members shape (M, *actual.shape), with M >= 1')
    if not np.isfinite(actual).all(): raise ValueError('Targets must be finite')
    with np.errstate(over='ignore',invalid='ignore'):
        centered=members-actual[None]
    valid=np.isfinite(centered).all(axis=0)
    safe=np.where(valid[None],centered,0.)
    count=members.shape[0]
    weights=(2*np.arange(1,count+1)-count-1).reshape((count,)+(1,)*actual.ndim)
    with np.errstate(over='ignore',invalid='ignore'):
        distance=np.abs(safe).mean(axis=0)
        half_dispersion=(np.sort(safe,axis=0)*weights).sum(axis=0)/(count*count)
        raw=distance-half_dispersion
        mean_error=np.abs(safe.mean(axis=0))
    valid &= np.isfinite(raw)&np.isfinite(distance)&np.isfinite(half_dispersion)
    mask=lambda value:np.where(valid,value,np.nan)
    return dict(empirical_crps_c=mask(np.maximum(raw,0.)),mean_member_abs_error_c=mask(distance),
                pair_half_dispersion_c=mask(half_dispersion),ensemble_mean_abs_error_c=mask(mean_error),valid=valid)


def empirical_crps(actual,members):
    """返回逐样本经验 CRPS，不跨变量/回合聚合，越小越好。"""
    return empirical_crps_components(actual,members)['empirical_crps_c']
