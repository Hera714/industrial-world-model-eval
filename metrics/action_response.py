"""配对动作响应诊断：方向、幅度和50%响应滞后；不把未达到阈值伪装成零滞后。"""
import numpy as np


def response_metrics(actual,predicted,onset_step,dt_minutes=10.,epsilon_c=.01):
    """actual/predicted 是计划减参考的状态差，索引0为动作执行前初态。

    动作索引 t 首先影响状态索引 t+1。滞后采用同一模拟器半峰值阈值，
    不为每个模型改用其自身峰值，以免弱响应模型仍显得时序准确。
    """
    actual,predicted=np.asarray(actual,float),np.asarray(predicted,float)
    if actual.ndim!=1 or actual.shape!=predicted.shape or not np.isfinite(actual).all():
        raise ValueError('Expected matching finite truth / predicted response vectors')
    if not 0<=onset_step<len(actual)-1 or dt_minutes<=0 or epsilon_c<=0:
        raise ValueError('Invalid onset or response metric settings')
    indices=np.arange(onset_step+1,len(actual))
    a,p=actual[indices],predicted[indices]; finite=np.isfinite(p)
    direction=finite&(np.abs(a)>=epsilon_c)
    row=dict(requested_steps=len(a),valid_steps=int(finite.sum()),valid_fraction=float(finite.mean()),
        direction_points=int(direction.sum()),direction_match_points=int((direction&(np.sign(p)==np.sign(a))).sum()),
        direction_accuracy=float((np.sign(p[direction])==np.sign(a[direction])).mean()) if direction.any() else np.nan,
        response_rmse_c=float(np.sqrt(np.mean((p[finite]-a[finite])**2))) if finite.any() else np.nan,
        true_peak_magnitude_c=float(np.abs(a).max()),peak_magnitude_error_c=np.nan,
        matched_peak_response_error_c=np.nan,terminal_response_error_c=np.nan,
        true_lag_steps=np.nan,predicted_lag_steps=np.nan,lag_error_minutes=np.nan,
        threshold_c=.5*float(np.abs(a).max()),lag_status='prediction_incomplete')
    peak=int(np.argmax(np.abs(a))); polarity=float(np.sign(a[peak]))
    if finite.all():
        row['peak_magnitude_error_c']=float(np.abs(p).max()-np.abs(a).max())
        row['matched_peak_response_error_c']=float(p[peak]-a[peak])
        row['terminal_response_error_c']=float(p[-1]-a[-1])
    if row['true_peak_magnitude_c']<epsilon_c:
        row['lag_status']='true_response_too_small'
        return row
    true_cross=indices[polarity*a>=row['threshold_c']][0]
    row['true_lag_steps']=int(true_cross-onset_step)
    if not finite.all(): return row
    crossing=indices[polarity*p>=row['threshold_c']]
    if not len(crossing):
        row['lag_status']='not_reached_within_horizon'
        return row
    row['predicted_lag_steps']=int(crossing[0]-onset_step)
    row['lag_error_minutes']=float((crossing[0]-true_cross)*dt_minutes)
    row['lag_status']='reached'
    return row
