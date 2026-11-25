# 总说明：对热工环境附加可记录的教学安全层；异常动作由保护层替换，环境本体仍严格拒绝非法动作。
import gymnasium as gym
from controllers.safety import SafetyGuard

class ThermalSafetyWrapper(gym.Wrapper):
    def __init__(self,env,low_c=12.0,high_c=30.0):
        super().__init__(env)
        self.guard=SafetyGuard(low_c,high_c,env.unwrapped.config.max_power_kw)
        self.finished=True; self.events=[]
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs); self.last_obs=obs; self.finished=False; self.events=[]
        return obs,info
    def step(self,action):
        if self.finished: raise RuntimeError('回合结束，必须重置')
        safe,decision=self.guard.apply(self.last_obs,action)
        obs,reward,terminated,truncated,info=self.env.step(safe)
        # 真实环境越过硬边界时终止；观测守卫可能无法阻止单步跨界。
        true_air=float(info['true_state_c'][0]); boundary=''
        if true_air<self.guard.low: boundary='true_low_boundary'
        elif true_air>self.guard.high: boundary='true_high_boundary'
        if boundary:
            terminated=True
            # 同一步达到时间上限时，按实际情况允许两个标记同时为真。
        info=dict(info); info.update(decision); info['environment_safety_reason']=boundary
        if decision['safety_reason'] or boundary:
            self.events.append({'step':info['step'],'safety_reason':decision['safety_reason'],'environment_safety_reason':boundary,'proposed_action_kw':decision['proposed_action_kw'],'executed_action_kw':info['action_kw']})
        self.last_obs=obs; self.finished=terminated or truncated
        return obs,reward,terminated,truncated,info
