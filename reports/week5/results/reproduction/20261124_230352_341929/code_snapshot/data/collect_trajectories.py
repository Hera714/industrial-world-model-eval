# 总说明：用随机分段动作和滞回控制采集完整回合，保留真实状态、带噪观测及动作时间。
import numpy as np
import pandas as pd
from dataclasses import asdict
import hashlib
import json
from simulators.simple_thermal import SimpleThermalEnv, ThermalConfig
from controllers.rule_based import HysteresisController

def collect_episodes(n_episodes=45, base_seed=1000, config=None, policies=('random', 'rule')):
    cfg=config or ThermalConfig(); rows=[]
    # 配置指纹区分不同物理参数；每条记录携带完整配置便于独立追溯。
    config_json = json.dumps(asdict(cfg), sort_keys=True)
    config_id = hashlib.sha256(config_json.encode()).hexdigest()[:16]
    for ep in range(n_episodes):
        env=SimpleThermalEnv(cfg); rng=np.random.default_rng(base_seed+10000+ep)
        init_air=float(rng.uniform(17,23)); init_wall=float(rng.uniform(17,22))
        obs,info=env.reset(seed=base_seed+ep,options={'initial_air_c':init_air,'initial_wall_c':init_wall})
        ctrl=HysteresisController(cfg.target_c,.5,cfg.max_power_kw); ctrl.reset()
        policy=policies[ep % len(policies)]; action=0.
        if policy == 'mpc':
            from controllers.mpc import RandomShootingMPC, MPCConfig
            ctrl = RandomShootingMPC(cfg, MPCConfig(seed=base_seed+10000+ep))
        elif policy not in ('random', 'rule'):
            raise ValueError('Unknown collection policy')
        for step in range(cfg.episode_steps):
            before=obs.copy()
            if policy=='random':
                if step%6==0: action=float(rng.uniform(0,cfg.max_power_kw))
                u=np.array([action],dtype=np.float32)
            elif policy == 'mpc': u, _ = ctrl.act(before)
            else: u=ctrl.act(before)
            obs,_,term,trunc,info=env.step(u)
            scenario_id = f'thermal_{config_id}_air{init_air:.8f}_wall{init_wall:.8f}'
            # 规范数组用 JSON 字符串存入 DataFrame，CSV 导出后可用 json.loads 恢复。
            transition = {
                'schema_version': '1.0',
                'episode_id': f'{scenario_id}_seed{base_seed+ep}_episode{ep}',
                'scenario_id': scenario_id, 'seed': base_seed+ep,
                'policy_seed': base_seed+10000+ep, 'source_step': step,
                'time': step*cfg.dt_hours, 'next_time': (step+1)*cfg.dt_hours,
                'dt': cfg.dt_hours, 'time_unit': 'hour', 'action_hold_steps': 1,
                'state': json.dumps(info['state_before_c'].tolist()),
                'observation': json.dumps(before.tolist()),
                'action_requested': json.dumps(u.tolist()),
                'action_applied': json.dumps([info['action_kw']]),
                'exogenous': json.dumps([info['outdoor_used_c'], info['internal_load_kw']]),
                'exogenous_available_at_decision': json.dumps([float(before[2]), info['internal_load_kw']]),
                'next_state': json.dumps(info['true_state_c'].tolist()),
                'next_observation': json.dumps(obs.tolist()),
                'process_noise': json.dumps(info['process_noise_c'].tolist()),
                'cost': info['cost'], 'energy_kwh': info['energy_kwh'],
                'config_id': config_id, 'environment_config': config_json,
                'end_reason': 'time_limit' if trunc else ('environment_terminal' if term else ''),
            }
            rows.append({**transition,'episode':ep,'step':step,'policy':policy,'environment_seed':base_seed+ep,
              'air':info['state_before_c'][0],'wall':info['state_before_c'][1],'outdoor':info['outdoor_used_c'],
              'observed_air':before[0],'observed_wall':before[1],'observed_outdoor':before[2],
              'action':info['action_kw'],'next_air':info['true_state_c'][0],'next_wall':info['true_state_c'][1],
              'terminated':term,'truncated':trunc})
            if term or trunc: break
    return pd.DataFrame(rows)
