"""热工启发的一步变化速率投影，保留原始输出和修正诊断。"""
import numpy as np
from models.dynamics_base import DynamicsModel
from simulators.simple_thermal import ThermalConfig


class RateConstrainedDynamics(DynamicsModel):
    """按允许功率和室外温度范围约束状态增量，内部负荷取配置常量。"""
    def __init__(self, model, outdoor_bounds=(-5.,17.), noise_margin=.03):
        super().__init__(*model.dimensions)
        self.model = model
        self.config = ThermalConfig()
        self.outdoor_bounds = outdoor_bounds
        self.noise_margin = noise_margin

    def predict_with_diagnostics(self, state, action, exogenous):
        """返回修正状态及原预测、增量边界和逐分量触发标记。"""
        x,u,d = self.inputs(state,action,exogenous)
        raw = self.model.predict_next(x,u,d)
        c = self.config
        air,wall = x[...,0],x[...,1]
        # 边界来自热交换项及实验允许的输入范围，每步保留 3σ 噪声余量。
        bounds = []
        for outside,power in ((self.outdoor_bounds[0],0.),(self.outdoor_bounds[1],c.max_power_kw)):
            air_delta = c.dt_hours/c.air_capacity*(c.air_wall_conductance*(wall-air)+c.air_out_conductance*(outside-air)+power+c.internal_load_kw)
            wall_delta = c.dt_hours/c.wall_capacity*(c.air_wall_conductance*(air-wall)+c.wall_out_conductance*(outside-wall))
            bounds.append(np.stack([air_delta,wall_delta],axis=-1))
        low,high = bounds[0]-self.noise_margin,bounds[1]+self.noise_margin
        delta = raw-x
        corrected = x+np.clip(delta,low,high)
        return corrected,dict(raw=raw,low_delta=low,high_delta=high,triggered=(delta<low)|(delta>high))

    def _predict(self,state,action,exogenous):
        return self.predict_with_diagnostics(state,action,exogenous)[0]
