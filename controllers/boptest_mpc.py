"""总说明：BOPTEST 历史世界模型的随机 shooting MPC，候选预算固定、天气保持当前、无有效轨迹则释放内置控制。"""
import time
import numpy as np
from models.boptest_dynamics import exogenous_features


def stage_cost(zone,energy,lower,upper,previous,action,dt=.25,comfort_weight=1.,energy_weight=1.,change_weight=.02):
    """舒适标准独立于可控设定值；舒适平方偏差°C²h、实耗kWh、动作变化°C²加权。"""
    violation=np.maximum(lower-zone,0)+np.maximum(zone-upper,0)
    return comfort_weight*violation**2*dt+energy_weight*energy+change_weight*(action-previous)**2


class BoptestShootingMPC:
    def __init__(self,model,config):self.model=model;self.config=config
    def act(self,history,timestamp,lower,upper,step,previous):
        """固定每步候选随机流，所有模型看到相同动作计划；计时检查不能抢占单次模型调用。"""
        start=time.perf_counter();p=self.config['control'];H=p['horizon'];N=p['candidates'];hold=p['action_hold_steps']
        rng=np.random.default_rng(604200+step);lo,hi=self.config['action']['experiment_bounds_c']
        actions=np.repeat(rng.uniform(lo,hi,size=(N,int(np.ceil(H/hold)))),hold,axis=1)[:,:H]
        # 各模型共享相同确定性候选，避免只靠随机搜索错过恒定设定值。
        for i,value in enumerate((lo,20.,21.,22.,hi)):actions[i]=value
        windows=np.repeat(np.asarray(history)[None],N,axis=0);costs=np.zeros(N);valid=np.ones(N,bool);prev=np.full(N,previous)
        weather=history[-1,6:8].copy();reason='';error=''
        try:
            for h in range(H):
                if time.perf_counter()-start>p['time_budget_s']:reason='timeout';break
                windows[:,-1,5]=actions[:,h]
                prediction=self.model.predict(windows)
                valid &= np.isfinite(prediction).all(1)&(prediction[:,0]>=0)&(prediction[:,0]<=50)&(prediction[:,5]>=0)&(prediction[:,5]<=10)
                costs+=stage_cost(prediction[:,0],prediction[:,5],lower[h],upper[h],prev,actions[:,h],
                    comfort_weight=p['comfort_weight'],energy_weight=p['energy_weight'],change_weight=p['action_change_weight'])
                prev=actions[:,h]
                if h+1<H:
                    future=exogenous_features(weather,np.asarray(timestamp+(h+1)*900))
                    nextrow=np.column_stack([prediction[:,:5],actions[:,h+1],np.repeat(future[None],N,axis=0)])
                    windows=np.concatenate([windows[:,1:],nextrow[:,None]],axis=1)
            if time.perf_counter()-start>p['time_budget_s']:reason='timeout'
            if not reason and not valid.any():reason='no_valid_candidates'
        except Exception as exc:reason='planning_error';error=repr(exc)
        elapsed=time.perf_counter()-start
        if reason:return None,dict(fallback_reason=reason,error=error,decision_time_s=elapsed,valid_candidates=int(valid.sum()))
        costs[~valid]=np.inf;idx=int(costs.argmin())
        return [float(actions[idx,0])],dict(fallback_reason='',error='',decision_time_s=elapsed,valid_candidates=int(valid.sum()),predicted_plan_cost=float(costs[idx]))
