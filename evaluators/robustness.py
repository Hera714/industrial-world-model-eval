"""总说明：按独立时间块计算天气支持与 bootstrap；不把重叠窗口或控制时间步作为独立重复。"""
import numpy as np


def weather_support(training,testing):
    """范围外比例仅是 OOD 证据之一，范围内不代表联合分布完全一致。"""
    train=np.asarray(training,float);test=np.asarray(testing,float)
    if train.ndim!=2 or test.ndim!=2 or train.shape[1]!=test.shape[1] or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError('Invalid weather arrays')
    lo=train.min(0);hi=train.max(0)
    return dict(lower=lo,upper=hi,outside_fraction=np.mean((test<lo)|(test>hi),axis=0))


def episode_bootstrap(values,seed=6042,repeats=2000):
    """输入必须已经是每个独立时间块一个配对差值；小样本区间只能作描述性参考。"""
    values=np.asarray(values,float)
    if values.ndim!=1 or len(values)==0 or not np.isfinite(values).all():raise ValueError('Invalid episode differences')
    rng=np.random.default_rng(seed);samples=values[rng.integers(0,len(values),size=(repeats,len(values)))].mean(1)
    return dict(mean=float(values.mean()),lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),
        episodes=len(values),seed=seed,repeats=repeats,warning='Descriptive small-sample interval; not timestep-independent inference')


def corrupt_boptest_observation(original, condition, rng, last_supply, scored_step):
    """温度高斯噪声或供水温度缺失；返回污染原值、因果填补值与逐通道缺失标记。"""
    original = np.asarray(original, dtype=float)
    raw = original.copy()
    missing = np.zeros(original.shape, dtype=bool)
    active = scored_step >= 0
    if active and condition == 'noise_0p3':
        raw[:3] += rng.normal(0, .3, size=3)
    if active and condition == 'missing_4' and scored_step % 24 < 4:
        missing[1] = True
        raw[1] = np.nan
    filled = raw.copy()
    if missing[1]:
        if last_supply is None:
            raise ValueError('Missing sensor has no prior valid observation')
        filled[1] = last_supply
    else:
        last_supply = filled[1]
    return raw, filled, missing, last_supply


def paired_metric_changes(frame, keys, condition_column, reference_condition, metrics):
    """按相同模型/日期/评测设置配对，输出变化减参考，拒绝重复身份与缺失参考。"""
    if frame.duplicated(keys + [condition_column]).any():
        raise ValueError('Repeated paired identities')
    reference = frame[frame[condition_column].eq(reference_condition)][keys + metrics]
    changed = frame[~frame[condition_column].eq(reference_condition)]
    paired = changed.merge(reference, on=keys, how='left', suffixes=('', '_reference'), validate='many_to_one', indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('Missing matched reference')
    paired = paired.drop(columns='_merge')
    for metric in metrics:
        paired[metric + '_delta'] = paired[metric] - paired[metric + '_reference']
        denominator = paired[metric + '_reference'].where(paired[metric + '_reference'].abs() > 1e-12)
        paired[metric + '_relative_percent'] = 100 * paired[metric + '_delta'] / denominator
    return paired
