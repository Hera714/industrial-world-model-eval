# 总说明：统一执行规则控制或 MPC 的真实模拟环境闭环并记录逐步观测、状态、扰动与代价。
import time
import numpy as np
from simulators.simple_thermal import SimpleThermalEnv, ThermalConfig
from controllers.rule_based import HysteresisController
from controllers.mpc import RandomShootingMPC, MPCConfig
from metrics.control import calculate_control_metrics

def run_episode(controller_name, scenario, env_config=None, mpc_config=None, dynamics=None):
    config=env_config or ThermalConfig(); env=SimpleThermalEnv(config)
    if controller_name=='rule':
        controller=HysteresisController(config.target_c,.5,config.max_power_kw)
    elif controller_name=='mpc':
        controller=RandomShootingMPC(config,mpc_config or MPCConfig(),dynamics=dynamics)
    else: raise ValueError('未知控制器')
    # 两个控制器只读带噪观测；每个场景使用相同种子和初始状态。
    obs,initial=env.reset(seed=scenario['seed'],options={'initial_air_c':scenario['initial_air_c'],'initial_wall_c':scenario['initial_wall_c']})
    controller.reset(); rows=[]; cumulative=0.
    for i in range(config.episode_steps):
        before=obs.copy(); start=time.perf_counter()
        decision=controller.act(before)
        elapsed=time.perf_counter()-start
        if controller_name=='mpc': action,detail=decision; reason=detail['fallback_reason'] or ''
        else: action=decision; reason=''
        if not env.action_space.contains(action): raise ValueError('控制器输出不合法')
        obs,reward,terminated,truncated,info=env.step(action)
        # 评测评分与环境一致：权重均为 1；与 MPC 内部规划目标区分。
        cumulative+=info['cost']
        rows.append({'step':i+1,'hour':(i+1)*config.dt_hours,'action_start_hour':i*config.dt_hours,
          'true_air_before_c':info['state_before_c'][0],'true_wall_before_c':info['state_before_c'][1],
          'observed_air_before_c':float(before[0]),'observed_wall_before_c':float(before[1]),'observed_outdoor_before_c':float(before[2]),
          'true_air_after_c':info['true_state_c'][0],'true_wall_after_c':info['true_state_c'][1],
          'observed_air_after_c':float(obs[0]),'observed_wall_after_c':float(obs[1]),'observed_outdoor_after_c':float(obs[2]),
          'outdoor_used_c':info['outdoor_used_c'],'outdoor_next_c':info['next_outdoor_c'],'internal_load_kw':info['internal_load_kw'],
          'action_kw':info['action_kw'],'target_c':info['target_c'],'energy_kwh':info['energy_kwh'],
          'tracking_cost':info['tracking_cost'],'energy_cost':info['energy_kwh'],'cost':info['cost'],'cumulative_cost':cumulative,
          'comfort_violation':info['comfort_violation'],'decision_time_s':elapsed,'fallback_reason':reason,
          'process_air_c':info['process_noise_c'][0],'process_wall_c':info['process_noise_c'][1],
          'measurement_air_c':info['measurement_noise_c'][0],'measurement_wall_c':info['measurement_noise_c'][1],
          'measurement_outdoor_c':info['measurement_noise_c'][2],'terminated':terminated,'truncated':truncated})
        if terminated or truncated: break
    if len(rows)!=config.episode_steps or not rows[-1]['truncated'] or rows[-1]['terminated']:
        raise RuntimeError(f'{controller_name} 未完成完整回合，已执行 {len(rows)} 步')
    summary=calculate_control_metrics(rows,config.dt_hours)
    if not np.isclose(summary['cumulative_cost'],cumulative): raise AssertionError('统一成本与环境成本不一致')
    return rows,summary,initial


def run_standard_episode(env, policy, seed, episode_id, make_row=None):
    """通用 Gymnasium 闭环采集，保留公开观测、原生奖励和诊断信息。"""
    obs, info = env.reset(seed=seed)
    rows = []
    names = ['setpoint', 'velocity', 'gain', 'shift', 'fatigue', 'consumption']
    while True:
        before = obs.copy()
        start = time.perf_counter()
        action = policy(before)
        if make_row is None:
            action = np.asarray(action, dtype=np.float32)
        elapsed = time.perf_counter() - start
        obs, reward, terminated, truncated, info = env.step(action)
        if make_row is not None:
            rows.append(make_row(obs, action, terminated, truncated, info))
        else:
            rows.append(dict(schema_version='1.0', episode_id=episode_id, seed=seed,
            step=info['step'], timestamp=None, time_unit='step',
            state=dict(zip(names, map(float, obs))), observation=obs.tolist(),
            observation_before=before.tolist(), action=action.tolist(),
            disturbance=dict(setpoint=float(obs[0]), noise=None),
            cost=float(info['cost']), reward=float(reward),
            constraints=dict(action_valid=bool(env.action_space.contains(action)),
                observation_in_declared_space=bool(env.observation_space.contains(obs))),
            terminated=terminated, truncated=truncated, done=terminated or truncated,
            end_reason=info.get('end_reason'), upstream_info=info['upstream_info'],
            decision_time_s=elapsed))
        if terminated or truncated:
            break
    return rows


"""总说明：BOPTEST 真实闭环对照，重放同一初始上下文；只改模型，保持规划预算和独立舒适标准。"""
from pathlib import Path
import sys,json,time,hashlib,os
import numpy as np
import pandas as pd
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulators.boptest_control import BoptestControlEnv,measurement_state
from controllers.boptest_mpc import stage_cost
from controllers.mpc import create_boptest_mpc
from models.boptest_dynamics import BoptestHistoryModel,exogenous_features


from metrics.control import calculate_boptest_control_metrics

def run_boptest_episode(cfg,episode,spec,condition='clean'):
    """服务异常保存部分轨迹并报错；只有完整回合才写 complete 清单，便于可靠续跑。"""
    base=ROOT/cfg['output'];output=base/'closed_loop';output.mkdir(parents=True,exist_ok=True)
    name=spec['name'];runid=f'{episode["episode_id"]}__{name}__{condition}';path=output/(runid+'.csv');meta=path.with_suffix('.json')
    if path.exists() and meta.exists() and json.loads(meta.read_text()).get('status')=='complete':
        saved=json.loads(meta.read_text())
        if saved.get('checkpoint_sha256')!=spec.get('sha256'):raise ValueError('Stale control checkpoint: '+runid)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=saved['csv_sha256']:raise ValueError('Control CSV changed: '+runid)
        print('skip '+runid,flush=True);return
    local=dict(cfg,start_time=episode['start_day']*86400,episode_steps=cfg['context_steps']+cfg['episode_steps'])
    recorded=pd.read_csv(base/'data'/(episode['episode_id']+'.csv'))
    controller=None if name=='embedded' else create_boptest_mpc(spec,cfg,ROOT)
    env=BoptestControlEnv(local);rows=[];history=[];rng=np.random.default_rng(64042+episode['start_day']);last_supply=None
    try:
        _,initial=env.reset(seed=42);points=initial['measurements'];previous=float(recorded.action_applied_c.iloc[0])
        forecast=env.request('PUT',f'/forecast/{env.testid}',dict(point_names=['TDryBul','HGloHor','LowerSetp[1]','UpperSetp[1]'],
            horizon=(local['episode_steps']+cfg['control']['horizon'])*900,interval=900))
        (output/(runid+'_forecast.json')).write_text(json.dumps(forecast))
        for k in range(local['episode_steps']):
            timestamp=local['start_time']+k*900;true=measurement_state(points);observed=true.copy()
            if k>=cfg['context_steps'] and condition=='noise_0p3':observed[:3]+=rng.normal(0,.3,3)
            missing=condition=='missing_4' and k>=cfg['context_steps'] and (k-cfg['context_steps'])%24<4
            if missing and last_supply is not None:observed[1]=last_supply
            else:last_supply=observed[1]
            weather=np.array([np.interp(timestamp,forecast['time'],forecast['TDryBul'])-273.15,np.interp(timestamp,forecast['time'],forecast['HGloHor'])])
            exo=exogenous_features(weather,np.asarray(timestamp))
            history.append(np.r_[observed,previous,exo]);history=history[-cfg['history_steps']:]
            detail=dict(decision_time_s=0.,fallback_reason='',valid_candidates=None)
            if k<cfg['context_steps']:action=[float(recorded.action_applied_c.iloc[k])]
            elif controller is None:action=None
            else:
                future=timestamp+np.arange(1,cfg['control']['horizon']+1)*900
                shift=1. if condition=='goal_plus1' else 0.
                lower=np.interp(future,forecast['time'],forecast['LowerSetp[1]'])-273.15+shift
                upper=np.interp(future,forecast['time'],forecast['UpperSetp[1]'])-273.15+shift
                action,detail=controller.act(np.asarray(history),timestamp,lower,upper,k,previous)
            _,_,term,trunc,info=env.safe_step(action,detail['fallback_reason']) if controller is not None and k>=cfg['context_steps'] else env.step(action)
            after=measurement_state(info['measurements']);actual_action=info['applied_action_c']
            history[-1][5]=actual_action
            shift=1. if condition=='goal_plus1' else 0.
            tnext=timestamp+900;lower=float(np.interp(tnext,forecast['time'],forecast['LowerSetp[1]'])-273.15+shift)
            upper=float(np.interp(tnext,forecast['time'],forecast['UpperSetp[1]'])-273.15+shift)
            violation=max(lower-after[0],0)+max(after[0]-upper,0)
            p=cfg['control'];cost=float(stage_cost(after[0],info['energy_kwh'],lower,upper,previous,actual_action,
                comfort_weight=p['comfort_weight'],energy_weight=p['energy_weight'],change_weight=p['action_change_weight']))
            rows.append(dict(run_id=runid,episode_id=episode['episode_id'],split=episode['split'],model=name,family=spec.get('kind',name),training_seed=spec.get('seed',-1),
                condition=condition,step=k,timestamp=timestamp,context=k<cfg['context_steps'],true_before=json.dumps(true.tolist()),observed_before=json.dumps(observed.tolist()),
                true_after=json.dumps(after.tolist()),weather=json.dumps(weather.tolist()),requested_action_c=action[0] if action else None,applied_action_c=actual_action,
                override_active=info['override_active'],missing_supply=missing,lower_c=lower,upper_c=upper,zone_c=after[0],
                energy_kwh=info['energy_kwh'],comfort_violation_c=violation,comfort_violation=violation>0,cost=cost,
                tracking_error_c=abs(after[0]-(lower+upper)/2),decision_time_s=detail['decision_time_s'],fallback_reason=info['fallback_reason'],
                valid_candidates=detail.get('valid_candidates'),native_kpi=json.dumps(info['native_kpi']),terminated=term,truncated=trunc))
            if k==cfg['context_steps']-1:context_kpi=info['native_kpi'];context_energy=env.last_energy
            previous=actual_action;points=info['measurements']
        assert rows[-1]['truncated'] and not rows[-1]['terminated']
        frame=pd.DataFrame(rows);frame.to_csv(path,index=False);scored=frame[~frame.context]
        reasons=scored.fallback_reason.value_counts().to_dict();reasons.pop('',None)
        energy=float(scored.energy_kwh.sum());np.testing.assert_allclose(energy,env.last_energy-context_energy,atol=1e-8)
        metric=dict(status='complete',run_id=runid,episode_id=episode['episode_id'],split=episode['split'],model=name,family=spec.get('kind',name),condition=condition,
            training_seed=spec.get('seed',-1),steps=len(scored),cost=float(scored.cost.sum()),energy_kwh=energy,
            comfort_violation_rate=float(scored.comfort_violation.mean()),comfort_degree_hours=float(scored.comfort_violation_c.sum()*.25),
            center_tracking_mae_c=float(scored.tracking_error_c.mean()),decision_mean_ms=float(scored.decision_time_s.mean()*1000),
            decision_p95_ms=float(scored.decision_time_s.quantile(.95)*1000),fallback_count=int(scored.fallback_reason.ne('').sum()),fallback_reasons=reasons,
            native_context_kpi=context_kpi,native_final_kpi=info['native_kpi'],
            native_ener_delta_kwh_m2=(info['native_kpi']['ener_tot']-context_kpi['ener_tot']),
            warning='Native final KPI covers context+evaluation; report differences only for additive cumulative quantities, never subtract maxima or ratios.',
            csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_sha256=spec.get('sha256'))
        metric.update(calculate_boptest_control_metrics(scored))
        meta.write_text(json.dumps(metric,indent=2));print('completed '+runid+f' cost={metric["cost"]:.3f}',flush=True)
    except Exception as exc:
        if rows:pd.DataFrame(rows).to_csv(path.with_suffix('.partial.csv'),index=False)
        (output/(runid+'_failure.json')).write_text(json.dumps(dict(status='failed',error=repr(exc),completed_steps=len(rows),run_id=runid)))
        raise
    finally:env.close()

