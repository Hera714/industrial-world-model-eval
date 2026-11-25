# 总说明：对完整控制轨迹统一计算环境成本、跟踪、能耗、舒适性和决策耗时。
from collections import Counter
import numpy as np

def calculate_control_metrics(rows, dt_hours, tracking_weight=1.0, energy_weight=1.0):
    """统一评分：真实下一室温跟踪误差平方 + 实际功率乘步长。"""
    if not rows or not np.isfinite([dt_hours,tracking_weight,energy_weight]).all() or dt_hours<=0 or min(tracking_weight,energy_weight)<0:
        raise ValueError('轨迹或成本配置无效')
    temp=np.array([r['true_air_after_c'] for r in rows],float)
    target=np.array([r['target_c'] for r in rows],float)
    power=np.array([r['action_kw'] for r in rows],float)
    durations=np.array([r['decision_time_s'] for r in rows],float)
    if not all(np.isfinite(x).all() for x in (temp,target,power,durations)) or (power<0).any() or (durations<0).any():
        raise ValueError('轨迹包含无效温度、动作或耗时')
    energy=power*dt_hours; tracking=(temp-target)**2
    reasons=Counter(r['fallback_reason'] for r in rows if r['fallback_reason'])
    return {'steps':len(rows),'tracking_mae_c':float(np.abs(temp-target).mean()),
      'energy_kwh':float(energy.sum()),'tracking_cost':float(tracking.sum()),
      'energy_cost':float(energy.sum()*energy_weight),
      'cumulative_cost':float((tracking_weight*tracking+energy_weight*energy).sum()),
      'comfort_violation_rate':float(np.mean([r['comfort_violation'] for r in rows])),
      'decision_mean_ms':float(durations.mean()*1000),'decision_p95_ms':float(np.quantile(durations,.95)*1000),
      'fallback_count':sum(reasons.values()),'fallback_reasons':dict(reasons)}


def calculate_boptest_control_metrics(frame):
    """对已排除上下文的BOPTEST真实轨迹计算KPI，能耗来自原生积分增量。"""
    required = ['cost', 'energy_kwh', 'comfort_violation', 'comfort_violation_c',
                'tracking_error_c', 'decision_time_s', 'fallback_reason']
    if frame.empty or any(name not in frame for name in required):
        raise ValueError('Incomplete BOPTEST scoring trajectory')
    numeric = frame[['cost','energy_kwh','comfort_violation_c','tracking_error_c','decision_time_s']].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError('Nonfinite BOPTEST scoring data')
    reasons = frame.fallback_reason.fillna('')
    return dict(steps=len(frame), cost=float(frame.cost.sum()), energy_kwh=float(frame.energy_kwh.sum()),
        comfort_violation_rate=float(frame.comfort_violation.mean()),
        comfort_degree_hours=float(frame.comfort_violation_c.sum() * .25),
        center_tracking_mae_c=float(frame.tracking_error_c.mean()),
        decision_mean_ms=float(frame.decision_time_s.mean() * 1000),
        decision_p95_ms=float(frame.decision_time_s.quantile(.95) * 1000),
        fallback_count=int(reasons.ne('').sum()), fallback_reasons=reasons[reasons.ne('')].value_counts().to_dict())
