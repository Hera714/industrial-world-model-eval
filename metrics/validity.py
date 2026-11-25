"""预测有效性检查：逐变量记录异常、幅度与停止标记，只诊断、不投影或修正输出。"""
import numpy as np


def thermal_rate_bounds(state, rate_config, physics):
    """用冻结输入范围和当前模型输入状态推导单步增量上下界，单位 °C/step。"""
    state = np.asarray(state, dtype=float)
    air, wall = state[..., 0], state[..., 1]
    bounds = []
    for outside, power in zip(rate_config['outdoor_bounds_c'], rate_config['power_bounds_kw']):
        exchange = physics['air_wall_conductance']*(wall-air)
        da = physics['dt_hours']/physics['air_capacity']*(exchange+
            physics['air_out_conductance']*(outside-air)+power+rate_config['load_kw'])
        dw = physics['dt_hours']/physics['wall_capacity']*(-exchange+
            physics['wall_out_conductance']*(outside-wall))
        bounds.append(np.stack([da, dw], axis=-1))
    margin = rate_config['noise_margin_c']
    return bounds[0]-margin, bounds[1]+margin


def check_prediction(previous, predicted, config, physics=None):
    """检查同形状态数组；非有限/数值发散停止，合理性和速率异常仅标记继续。"""
    previous, predicted = np.asarray(previous, dtype=float), np.asarray(predicted, dtype=float)
    if previous.shape != predicted.shape or previous.shape[-1] != 2:
        raise ValueError('Validity expects matching (...,2) thermal states')
    finite = np.isfinite(predicted)
    events = {'nan': np.isnan(predicted), 'inf': np.isinf(predicted)}
    details = {}
    for name, key in [('observation_bounds', 'observation_bounds_c'),
                      ('plausibility_bounds', 'plausibility_bounds_c'),
                      ('divergence', 'divergence_bounds_c')]:
        low, high = config.get(key, [None, None])
        low = -np.inf if low is None else low
        high = np.inf if high is None else high
        events[name] = finite & ((predicted < low) | (predicted > high))
        with np.errstate(invalid='ignore'):
            magnitude = np.where(finite, np.maximum(np.maximum(low-predicted, predicted-high), 0), np.nan)
        details[name] = dict(low=np.broadcast_to(low, predicted.shape), high=np.broadcast_to(high, predicted.shape),
                             magnitude=magnitude)
    if physics is not None and config.get('rate_check'):
        low, high = thermal_rate_bounds(previous, config['rate_check'], physics)
        delta = predicted-previous
        # 1e-12 仅抵消状态加减的浮点舍入，不改变 0.03 °C 的物理余量。
        events['rate'] = finite & ((delta < low-1e-12) | (delta > high+1e-12))
        details['rate'] = dict(low=low, high=high, magnitude=np.maximum(np.maximum(low-delta, delta-high), 0))
    else:
        events['rate'] = np.zeros_like(finite)
    stop = (events['nan'] | events['inf'] | events['divergence']).any(axis=-1)
    return dict(events=events, details=details, stop=stop,
                any_anomaly=np.logical_or.reduce(list(events.values())).any(axis=-1))
