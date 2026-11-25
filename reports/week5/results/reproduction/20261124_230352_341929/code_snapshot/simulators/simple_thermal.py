# 总说明：实现 Gymnasium 二阶热工环境，模拟加热动作、天气扰动、状态转移和噪声。
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, asdict

# 参数使用小时、kW、kWh/°C 和 kW/°C；配置创建后保持不变。
@dataclass(frozen=True)
class ThermalConfig:
    dt_hours: float = 1/6
    episode_steps: int = 144
    max_power_kw: float = 5.0
    target_c: float = 21.0
    initial_air_c: float = 18.0
    initial_wall_c: float = 19.0
    outdoor_mean_c: float = 6.0
    outdoor_amplitude_c: float = 4.0
    internal_load_kw: float = 0.3
    air_capacity: float = 2.0
    wall_capacity: float = 12.0
    air_wall_conductance: float = 0.3
    air_out_conductance: float = 0.15
    wall_out_conductance: float = 0.1
    weather_noise_std: float = 0.3
    process_noise_std: float = 0.01
    measurement_noise_std: float = 0.05
    comfort_low_c: float = 20.0
    comfort_high_c: float = 23.0
    # 校验参数范围和显式离散更新的稳定性条件。
    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()): raise ValueError('Nonfinite configuration')
        if self.dt_hours<=0 or not isinstance(self.episode_steps,int) or self.episode_steps<1 or self.max_power_kw<=0 or min(self.air_capacity,self.wall_capacity)<=0: raise ValueError('Invalid positive parameter')
        if min(self.air_wall_conductance,self.air_out_conductance,self.wall_out_conductance,self.weather_noise_std,self.process_noise_std,self.measurement_noise_std)<0 or self.comfort_low_c>=self.comfort_high_c: raise ValueError('Invalid bounds or noise')
        if self.dt_hours*(self.air_wall_conductance+self.air_out_conductance)/self.air_capacity>1 or self.dt_hours*(self.air_wall_conductance+self.wall_out_conductance)/self.wall_capacity>1: raise ValueError('Unstable time step')

# 环境通过 reset 初始化，通过 step 执行一个控制时段。
class SimpleThermalEnv(gym.Env):
    """Action applies over [t,t+1). Comfort is soft; time limit truncates."""
    metadata={'render_modes':[]}
    def __init__(self,config=None):
        super().__init__(); self.config=config or ThermalConfig()
        self.action_space=spaces.Box(0,self.config.max_power_kw,(1,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(3,),dtype=np.float32); self._done=True
    # 生成具有日周期的室外温度，并加入天气扰动。
    def _outdoor(self,step):
        c=self.config
        return float(c.outdoor_mean_c+c.outdoor_amplitude_c*np.sin(2*np.pi*(step*c.dt_hours-8)/24)+self.weather_rng.normal(0,c.weather_noise_std))
    # 测量噪声仅叠加到观测，不改变真实状态。
    def _observe(self):
        noise=self.measurement_rng.normal(0,self.config.measurement_noise_std,3)
        return (np.array([*self.state,self.outdoor])+noise).astype(np.float32),noise
    # 从环境随机源派生天气、过程噪声和测量噪声的独立随机流。
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        seeds=self.np_random.integers(0,np.iinfo(np.uint64).max,3,dtype=np.uint64)
        self.weather_rng,self.process_rng,self.measurement_rng=(np.random.default_rng(int(x)) for x in seeds)
        options=options or {}; c=self.config
        if set(options)-{'initial_air_c','initial_wall_c'}: raise ValueError('Unknown reset option')
        self.state=np.array([options.get('initial_air_c',c.initial_air_c),options.get('initial_wall_c',c.initial_wall_c)],dtype=np.float64)
        if not np.isfinite(self.state).all(): raise ValueError('Nonfinite initial state')
        self.step_count=0; self._done=False; self.outdoor=self._outdoor(0); obs,noise=self._observe()
        return obs,{'true_state_c':self.state.copy(),'outdoor_c':self.outdoor,'measurement_noise_c':noise.copy(),'target_c':c.target_c,'step':0}
    # 动作作用于当前时刻到下一时刻；无效动作在状态更新前拒绝。
    def step(self,action):
        if self._done: raise RuntimeError('Call reset before step or after episode end')
        a=np.asarray(action,dtype=np.float64); c=self.config
        if a.shape!=(1,) or not np.isfinite(a).all() or not 0<=a[0]<=c.max_power_kw: raise ValueError('Action must be one finite value in range')
        power=float(a[0]); air,wall=self.state; outdoor=self.outdoor
        # 根据室内、墙体和室外之间的热交换更新真实状态。
        process=self.process_rng.normal(0,c.process_noise_std,2); exchange=c.air_wall_conductance*(wall-air)
        na=air+c.dt_hours/c.air_capacity*(exchange+c.air_out_conductance*(outdoor-air)+power+c.internal_load_kw)+process[0]
        nw=wall+c.dt_hours/c.wall_capacity*(-exchange+c.wall_out_conductance*(outdoor-wall))+process[1]
        self.state=np.array([na,nw]); self.step_count+=1; self.outdoor=self._outdoor(self.step_count)
        # 更新观测并记录示例成本；能耗按功率乘步长计算。
        obs,measurement=self._observe(); energy=power*c.dt_hours; tracking=(na-c.target_c)**2; cost=tracking+energy
        # 舒适区不触发终止；到达回合长度时仅触发截断。
        truncated=self.step_count>=c.episode_steps; self._done=truncated
        # 返回真实状态、噪声和扰动，供实验日志与评测使用。
        info={'step':self.step_count,'state_before_c':np.array([air,wall]),'true_state_c':self.state.copy(),'action_kw':power,'outdoor_used_c':outdoor,'next_outdoor_c':self.outdoor,'internal_load_kw':c.internal_load_kw,'process_noise_c':process.copy(),'measurement_noise_c':measurement.copy(),'target_c':c.target_c,'energy_kwh':energy,'tracking_cost':tracking,'cost':cost,'comfort_violation':bool(na<c.comfort_low_c or na>c.comfort_high_c)}
        return obs,-float(cost),False,truncated,info
