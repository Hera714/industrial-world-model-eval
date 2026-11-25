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
