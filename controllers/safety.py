# 总说明：教学用观测安全保护与动作审查，不代表真实设备安全控制。
import numpy as np

class SafetyGuard:
    def __init__(self, low_c=12.0, high_c=30.0, max_power_kw=5.0):
        if not np.isfinite([low_c,high_c,max_power_kw]).all() or low_c>=high_c or max_power_kw<=0: raise ValueError('安全边界无效')
        self.low=float(low_c); self.high=float(high_c); self.max_power=float(max_power_kw)
    def apply(self, observation, action):
        # 观测非法时选择零功率；温度越界采用简化的低温全开、高温关断。
        try: obs=np.asarray(observation,dtype=float)
        except (ValueError,TypeError): obs=np.array([np.nan])
        if obs.shape!=(3,) or not np.isfinite(obs).all(): return np.array([0.],np.float32),{'safety_reason':'invalid_observation','proposed_action_kw':None}
        try: a=np.asarray(action,dtype=float)
        except (ValueError,TypeError): a=np.array([np.nan])
        proposed=float(a[0]) if a.shape==(1,) and np.isfinite(a[0]) else None
        if proposed is None: return np.array([0.],np.float32),{'safety_reason':'invalid_action','proposed_action_kw':None}
        if obs[0]<self.low: return np.array([self.max_power],np.float32),{'safety_reason':'low_temperature','proposed_action_kw':proposed}
        if obs[0]>self.high: return np.array([0.],np.float32),{'safety_reason':'high_temperature','proposed_action_kw':proposed}
        bounded=float(np.clip(proposed,0,self.max_power))
        return np.array([bounded],np.float32),{'safety_reason':'action_clipped' if bounded!=proposed else '', 'proposed_action_kw':proposed}
