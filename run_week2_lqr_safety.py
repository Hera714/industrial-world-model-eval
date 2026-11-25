# 总说明：配对比较受限 LQR 与 MPC，并保存故障注入与安全层实际动作日志。
import csv, json, time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulators.simple_thermal import SimpleThermalEnv, ThermalConfig
from simulators.safety import ThermalSafetyWrapper
from controllers.lqr import ThermalLQR
from controllers.mpc import RandomShootingMPC, MPCConfig
from metrics.control import calculate_control_metrics

def save(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def main():
    out=Path('reports/week2/results/lqr_safety'); out.mkdir(parents=True,exist_ok=True)
    cfg=ThermalConfig(); lqr=ThermalLQR(cfg); mpc_config=MPCConfig(); summary=[]; cases=[]
    metadata={'thermal':asdict(cfg),'lqr':{'A':lqr.A.tolist(),'B':lqr.B.tolist(),'Q':lqr.Q.tolist(),'R':lqr.R.tolist(),'K':lqr.K.tolist(),
      'outdoor_equilibrium_c':lqr.outdoor_equilibrium_c,'state_equilibrium_c':lqr.x_eq.tolist(),'action_equilibrium_kw':lqr.u_eq,
      'equilibrium_action_feasible':lqr.equilibrium_action_feasible,'note':'动作限幅后不是无约束 LQR 最优解；固定室外温度工作点，无天气前馈'},
      'safety':{'observed_low_c':12,'observed_high_c':30,'true_termination_low_c':12,'true_termination_high_c':30},
      'mpc':asdict(mpc_config),'seeds':[42,43,44]}
    (out/'config.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for air,wall in ((18,19),(20,20),(22,21)):
      for seed in (42,43,44):
        pair={}
        for kind in ('lqr','mpc'):
            env=ThermalSafetyWrapper(SimpleThermalEnv(cfg)); ctrl=ThermalLQR(cfg) if kind=='lqr' else RandomShootingMPC(cfg,mpc_config)
            obs,_=env.reset(seed=seed,options={'initial_air_c':air,'initial_wall_c':wall}); ctrl.reset(); rows=[]
            for step in range(cfg.episode_steps):
                old=obs.copy(); t=time.perf_counter(); action,decision=ctrl.act(old); elapsed=time.perf_counter()-t
                obs,reward,terminated,truncated,info=env.step(action)
                rows.append({'step':step+1,'hour':(step+1)*cfg.dt_hours,'observed_air_before_c':float(old[0]),'observed_wall_before_c':float(old[1]),'observed_outdoor_before_c':float(old[2]),
                  'true_air_after_c':info['true_state_c'][0],'true_wall_after_c':info['true_state_c'][1],'outdoor_used_c':info['outdoor_used_c'],
                  'process_air_c':info['process_noise_c'][0],'process_wall_c':info['process_noise_c'][1],
                  'measurement_air_c':info['measurement_noise_c'][0],'measurement_wall_c':info['measurement_noise_c'][1],'measurement_outdoor_c':info['measurement_noise_c'][2],
                  'target_c':info['target_c'],'action_kw':info['action_kw'],'raw_action_kw':decision.get('raw_action_kw',float(action[0])),
                  'lqr_action_clipped':decision.get('action_clipped',False),'safety_reason':info['safety_reason'],
                  'environment_safety_reason':info['environment_safety_reason'],'fallback_reason':decision.get('fallback_reason') or '',
                  'error_message':decision.get('error_message') or '', 'energy_kwh':info['energy_kwh'],'cost':info['cost'],
                  'comfort_violation':info['comfort_violation'],'decision_time_s':elapsed,'terminated':terminated,'truncated':truncated})
                if terminated or truncated: break
            pair[kind]=rows; m=calculate_control_metrics(rows,cfg.dt_hours)
            summary.append({'initial_air_c':air,'seed':seed,'controller':kind,**{k:v for k,v in m.items() if k!='fallback_reasons'},
               'lqr_clipped_count':sum(r['lqr_action_clipped'] for r in rows),'safety_interventions':len(env.events),'terminated':terminated,'truncated':truncated})
            save(out/f'air{air}_seed{seed}_{kind}.csv',rows)
        # 回合未提前终止时检查配对扰动；安全终止时保留并报告实际长度。
        overlap=min(len(pair['lqr']),len(pair['mpc']))
        for key in ('outdoor_used_c','process_air_c','process_wall_c','measurement_air_c','measurement_wall_c','measurement_outdoor_c'):
            np.testing.assert_array_equal([r[key] for r in pair['lqr'][:overlap]],[r[key] for r in pair['mpc'][:overlap]])
        cases.append(pair)
    save(out/'episode_metrics.csv',summary)
    # 在独立短回合中注入越界与非法动作；不混入正常性能评测。
    events=[]
    for name,init,action in [('high',31,[5]),('low',5,[0]),('invalid',21,[float('nan')]),('clip',21,[8])]:
        env=ThermalSafetyWrapper(SimpleThermalEnv(cfg)); obs,_=env.reset(seed=8,options={'initial_air_c':init})
        _,_,term,trunc,info=env.step(action)
        events.append({'case':name,'executed_action_kw':info['action_kw'],'safety_reason':info['safety_reason'],'environment_safety_reason':info['environment_safety_reason'],'terminated':term})
    m=RandomShootingMPC()
    with patch.object(m,'_plan',return_value=(None,'no_valid_candidates')): action,info=m.act([19,19,6])
    events.append({'case':'no_valid_candidates','executed_action_kw':float(action[0]),'safety_reason':info['fallback_reason'],'environment_safety_reason':'','terminated':False})
    ticks=iter([0.0,2.0,2.0])
    timed=RandomShootingMPC(config=MPCConfig(time_budget_s=1),clock=lambda:next(ticks))
    action,info=timed.act([19,19,6])
    events.append({'case':'timeout','executed_action_kw':float(action[0]),'safety_reason':info['fallback_reason'],'environment_safety_reason':'','terminated':False})
    class Broken:
        def predict_next(self,*args): raise RuntimeError('injected model failure')
    action,info=RandomShootingMPC(dynamics=Broken()).act([19,19,6])
    events.append({'case':'model_exception','executed_action_kw':float(action[0]),'safety_reason':info['fallback_reason'],'environment_safety_reason':'','terminated':False})
    (out/'failure_injection.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata['lqr'],indent=2)); print(json.dumps(events,ensure_ascii=False,indent=2))
    for kind in ('lqr','mpc'):
        items=[r for r in summary if r['controller']==kind]
        print(kind,'mean_cost',np.mean([r['cumulative_cost'] for r in items]),'mean_MAE',np.mean([r['tracking_mae_c'] for r in items]),'clipped_total',sum(r['lqr_clipped_count'] for r in items),'terminated',sum(r['terminated'] for r in items))
if __name__=='__main__': main()
