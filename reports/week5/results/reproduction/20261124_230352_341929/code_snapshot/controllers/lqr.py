# 总说明：对二阶热工系统构造离散 LQR，记录工作点与未经限幅的动作。
import numpy as np
from scipy.linalg import solve_discrete_are
from simulators.simple_thermal import ThermalConfig

class ThermalLQR:
    def __init__(self, thermal=None, outdoor_equilibrium_c=6.0, q_air=1.0, q_wall=0.1, r=0.1):
        self.c=thermal or ThermalConfig(); c=self.c
        if not np.isfinite([outdoor_equilibrium_c,q_air,q_wall,r]).all() or q_air<0 or q_wall<0 or r<=0: raise ValueError('LQR 参数无效')
        self.outdoor_equilibrium_c=float(outdoor_equilibrium_c)
        # 模型本身为线性仿射系统，围绕室温目标和固定室外温度求稳态。
        ga,go,gw=c.air_wall_conductance,c.air_out_conductance,c.wall_out_conductance
        if ga+gw<=0: raise ValueError('墙体稳态不可唯一确定')
        wall=(ga*c.target_c+gw*outdoor_equilibrium_c)/(ga+gw)
        power=ga*(c.target_c-wall)+go*(c.target_c-outdoor_equilibrium_c)-c.internal_load_kw
        self.x_eq=np.array([c.target_c,wall]); self.u_eq=float(power)
        self.A=np.array([[1-c.dt_hours*(ga+go)/c.air_capacity,c.dt_hours*ga/c.air_capacity],[c.dt_hours*ga/c.wall_capacity,1-c.dt_hours*(ga+gw)/c.wall_capacity]])
        self.B=np.array([[c.dt_hours/c.air_capacity],[0.]])
        self.Q=np.diag([q_air,q_wall]); self.R=np.array([[r]])
        self.P=solve_discrete_are(self.A,self.B,self.Q,self.R)
        self.K=np.linalg.solve(self.R+self.B.T@self.P@self.B,self.B.T@self.P@self.A)
        self.equilibrium_action_feasible=bool(0<=power<=c.max_power_kw)
    def reset(self): pass
    def act(self,observation):
        # 墙温与室温采用有噪观测；稳态天气固定，不读取未来真实天气。
        obs=np.asarray(observation,dtype=float)
        if obs.shape!=(3,) or not np.isfinite(obs).all(): raise ValueError('观测无效')
        raw=float(self.u_eq-(self.K@(obs[:2]-self.x_eq))[0]); action=float(np.clip(raw,0,self.c.max_power_kw))
        return np.array([action],dtype=np.float32),{'raw_action_kw':raw,'action_clipped':bool(raw!=action)}
