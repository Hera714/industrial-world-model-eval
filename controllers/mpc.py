# 总说明：随机 shooting MPC，使用带噪观测和无噪声热工模型规划，并提供可记录的规则回退。
from dataclasses import dataclass
import time
import numpy as np
from controllers.rule_based import HysteresisController
from simulators.simple_thermal import ThermalConfig

@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 12
    candidates: int = 256
    tracking_weight: float = 1.0
    energy_weight: float = 1.0
    action_change_weight: float = 0.02
    comfort_penalty_weight: float = 2.0
    terminal_weight: float = 0.0
    time_budget_s: float = 1.0
    seed: int = 123
    def __post_init__(self):
        if not isinstance(self.horizon,int) or isinstance(self.horizon,bool) or self.horizon<1 or not isinstance(self.candidates,int) or isinstance(self.candidates,bool) or self.candidates<1: raise ValueError('时域和候选数必须是正整数')
        values=[self.tracking_weight,self.energy_weight,self.action_change_weight,self.comfort_penalty_weight,self.terminal_weight,self.time_budget_s]
        if not np.isfinite(values).all() or min(values)<0 or self.time_budget_s<=0: raise ValueError('权重应非负且时间预算应为正')

class RandomShootingMPC:
    def __init__(self, thermal=None, config=None, clock=None, dynamics=None):
        self.thermal=thermal or ThermalConfig(); self.config=config or MPCConfig()
        self.dynamics=dynamics  # 可注入学习模型；None 表示环境已知无噪声方程。
        self.clock=clock or time.perf_counter
        self.fallback=HysteresisController(self.thermal.target_c,.5,self.thermal.max_power_kw)
        self.reset()
    def reset(self):
        # 每回合重置规划随机数和上一次实际输出动作。
        self.rng=np.random.default_rng(self.config.seed); self.previous_action=0.0; self.fallback.reset()
    def _plan(self, obs, start):
        c=self.thermal; p=self.config
        # 候选动作独立均匀采样，并固定当前观测室外温度为整个时域的天气预测。
        actions=self.rng.uniform(0,c.max_power_kw,size=(p.candidates,p.horizon))
        air=np.full(p.candidates,obs[0],dtype=float); wall=np.full(p.candidates,obs[1],dtype=float)
        outside=float(obs[2]); prev=np.full(p.candidates,self.previous_action); costs=np.zeros(p.candidates)
        for step in range(p.horizon):
            # 时间预算在每个向量化预测步前检查，不能抢占正在执行的 NumPy 调用。
            if self.clock()-start >= p.time_budget_s: return None,'timeout'
            u=actions[:,step]; exchange=c.air_wall_conductance*(wall-air)
            if self.dynamics is None:
                next_air=air+c.dt_hours/c.air_capacity*(exchange+c.air_out_conductance*(outside-air)+u+c.internal_load_kw)
                next_wall=wall+c.dt_hours/c.wall_capacity*(-exchange+c.wall_out_conductance*(outside-wall))
            else:
                predicted=self.dynamics.predict_next(np.column_stack([air,wall]),u,outside)
                if np.shape(predicted)!=(p.candidates,2): raise ValueError('动力学输出形状无效')
                next_air,next_wall=predicted[:,0],predicted[:,1]
            violation=np.maximum(c.comfort_low_c-next_air,0)+np.maximum(next_air-c.comfort_high_c,0)
            costs+=p.tracking_weight*(next_air-c.target_c)**2+p.energy_weight*u*c.dt_hours+p.action_change_weight*(u-prev)**2+p.comfort_penalty_weight*violation**2
            air,wall,prev=next_air,next_wall,u
        costs+=p.terminal_weight*(air-c.target_c)**2
        if self.clock()-start >= p.time_budget_s: return None,'timeout'
        valid=np.isfinite(costs)&np.isfinite(actions).all(axis=1)
        if not valid.any(): return None,'no_valid_candidates'
        costs[~valid]=np.inf
        return float(actions[np.argmin(costs),0]),None
    def act(self, observation):
        start=self.clock(); reason=None; error_message=None
        try:
            obs=np.asarray(observation,dtype=float)
            if obs.shape!=(3,) or not np.isfinite(obs).all():
                # 输入非法时无法让规则控制器安全判断，使用明确的零功率后备。
                action=0.0; reason='invalid_observation'
            else:
                with np.errstate(over='ignore',invalid='ignore'):
                    action,reason=self._plan(obs,start)
                if reason:
                    # 依据上一步实际动作初始化滞回记忆，再读取当前观测。
                    self.fallback.reset(heating_on=self.previous_action>0)
                    action=float(self.fallback.act(obs)[0])
                elif not np.isfinite(action) or not 0<=action<=self.thermal.max_power_kw:
                    reason='invalid_planned_action'
                    self.fallback.reset(heating_on=self.previous_action>0)
                    action=float(self.fallback.act(obs)[0])
        except Exception as exc:
            # 规划数值异常优先使用规则控制，观测非法时退到零功率。
            reason='planning_error:'+type(exc).__name__; error_message=str(exc)
            try:
                self.fallback.reset(heating_on=self.previous_action>0)
                action=float(self.fallback.act(observation)[0])
            except (ValueError,TypeError):
                action=0.0; reason='invalid_observation'
        elapsed=max(0.0,self.clock()-start)
        self.previous_action=action
        return np.array([action],dtype=np.float32),{'planning_time_s':elapsed,'fallback_reason':reason,'error_message':error_message}


def create_boptest_mpc(model_spec, config, root='.'):
    """公共入口：为BOPTEST历史模型创建相同预算的MPC，观测预测与区间能耗共同参与规划。"""
    from models.dynamics_base import load_boptest_dynamics
    from controllers.boptest_mpc import BoptestShootingMPC
    return BoptestShootingMPC(load_boptest_dynamics(model_spec, root), config)
