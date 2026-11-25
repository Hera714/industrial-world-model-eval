# 总说明：基于观测室温实现带滞回的开关控制；只输出加热功率，不读取真实状态。
import numpy as np

class HysteresisController:
    """低于下限开启，高于上限关闭，等于阈值或位于区间内保持状态。"""
    def __init__(self, target_c=21.0, deadband_c=0.5, heating_power_kw=5.0):
        if not np.isfinite([target_c,deadband_c,heating_power_kw]).all() or deadband_c<=0 or heating_power_kw<0:
            raise ValueError('控制参数必须有限，死区大于零，加热功率非负')
        self.target_c=float(target_c); self.deadband_c=float(deadband_c); self.heating_power_kw=float(heating_power_kw)
        self.reset()
    def reset(self, heating_on=False):
        # 每个新回合显式重置控制器的开关记忆。
        if not isinstance(heating_on, (bool, np.bool_)): raise ValueError('开关初始状态必须为布尔值')
        self.heating_on=bool(heating_on)
    def act(self, observation):
        # 观测顺序由热工环境定义：室温、墙温、室外温度。
        obs=np.asarray(observation,dtype=float)
        if obs.shape!=(3,) or not np.isfinite(obs).all(): raise ValueError('观测必须是三个有限数值')
        if obs[0] < self.target_c-self.deadband_c: self.heating_on=True
        elif obs[0] > self.target_c+self.deadband_c: self.heating_on=False
        return np.array([self.heating_power_kw if self.heating_on else 0.0],dtype=np.float32)
