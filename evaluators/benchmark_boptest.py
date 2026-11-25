"""总说明：统一 Runner 的 BOPTEST 回合与历史 rollout；复用环境适配器、控制成本及指标口径。"""
import json
import time
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError, URLError
import numpy as np
import pandas as pd
from simulators.boptest_control import BoptestControlEnv, measurement_state
from models.boptest_dynamics import exogenous_features
from controllers.boptest_mpc import stage_cost
from metrics.control import calculate_boptest_control_metrics
from metrics.prediction import error_by_variable
from models.dynamics_base import PredictionOutputError


class ConfiguredBoptestEnv(BoptestControlEnv):
    """沿用真实覆盖/释放/能耗积分接口；只读请求允许重试，推进及初始化禁止自动重试。"""
    def request(self, method, path, data=None):
        policy = self.config['failure_policy']
        read_only = method == 'GET' or (method == 'PUT' and path.startswith(('/forecast/', '/results/')))
        attempts = 1 + (policy['retries']['read_requests'] if read_only else 0)
        opener = build_opener(ProxyHandler({})) if self.config['url'].split('://', 1)[1].split(':')[0] in ('localhost', '127.0.0.1') else build_opener()
        for attempt in range(attempts):
            req = Request(self.config['url'].rstrip('/') + path,
                          None if data is None else json.dumps(data).encode(),
                          {'Content-Type': 'application/json'}, method=method)
            try:
                with opener.open(req, timeout=policy['timeouts']['http_seconds']) as response:
                    body = response.read()
                if path.startswith('/stop/'):
                    return None
                result = json.loads(body)
                if result.get('status', 200) != 200:
                    raise RuntimeError(f'BOPTEST response: {result}')
                return result.get('payload', result)
            except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
                retryable = not isinstance(error, HTTPError) or error.code >= 500
                if attempt + 1 == attempts or not retryable:
                    detail = error.read().decode(errors='replace') if isinstance(error, HTTPError) else str(error)
                    raise RuntimeError(f'BOPTEST {method} {path}: {detail}') from error


def corrupt_observation(original, condition, faults, rng, previous, step):
    """保留污染原值、因果填充值与缺失标记，故障参数全部来自配置。"""
    raw = np.asarray(original, float).copy()
    mask = np.zeros(5, bool)
    fault = faults.get(condition, {})
    if step >= 0 and fault.get('kind') == 'temperature_noise':
        raw[:3] += rng.normal(0, fault['std_degC'], 3)
    if step >= 0 and fault.get('kind') == 'sensor_missing' and step % fault['repeat_period_steps'] < fault['burst_steps']:
        mask[1] = True
        raw[1] = np.nan
    filled = raw.copy()
    if mask[1]:
        if previous is None:
            raise ValueError('Missing sensor has no causal history')
        filled[1] = previous
    else:
        previous = filled[1]
    return raw, filled, mask, previous


class ConfiguredMPC:
    """相同 shooting 机制，候选种子、动作界、权重和有效性阈值均取统一配置。"""
    def __init__(self, model, config):
        self.model, self.config = model, config

    def act(self, history, timestamp, lower, upper, step, previous):
        cfg = self.config
        p = cfg['horizon']
        horizon, count, hold = p['mpc_steps'], p['candidates'], p['action_hold_steps']
        start = time.perf_counter()
        reason, error = '', ''
        valid = np.ones(count, bool)
        try:
            lo, hi = cfg['env']['action']['bounds']
            rng = np.random.default_rng(cfg['seed']['candidate']['base'] + step)
            actions = np.repeat(rng.uniform(lo, hi, (count, int(np.ceil(horizon / hold)))), hold, axis=1)[:, :horizon]
            # 保留 Week6 的确定性候选，并在配置改变动作界时确保候选合法。
            for i, value in enumerate((lo, 20., 21., 22., hi)):
                actions[i] = np.clip(value, lo, hi)
            windows = np.repeat(history[None], count, axis=0)
            costs, prev = np.zeros(count), np.full(count, previous)
            weather = history[-1, 6:8].copy()
            weights = cfg['metrics']['cost_weights']
            for k in range(horizon):
                if time.perf_counter() - start >= cfg['failure_policy']['timeouts']['planning_seconds']:
                    reason = 'timeout'
                    break
                windows[:, -1, 5] = actions[:, k]
                prediction = self.model.predict(windows)
                validity = cfg['metrics']['validity']
                good = np.isfinite(prediction).all(axis=1)
                good &= (prediction[:, 0] >= validity['zone_degC'][0]) & (prediction[:, 0] <= validity['zone_degC'][1])
                good &= (prediction[:, 5] >= validity['interval_energy_kwh'][0]) & (prediction[:, 5] <= validity['interval_energy_kwh'][1])
                valid &= good
                costs += stage_cost(prediction[:, 0], prediction[:, 5], lower[k], upper[k], prev, actions[:, k],
                                    comfort_weight=weights['comfort'], energy_weight=weights['energy'], change_weight=weights['action_change'])
                prev = actions[:, k]
                if k + 1 < horizon:
                    future = exogenous_features(weather, np.asarray(timestamp + (k + 1) * 900))
                    nextrow = np.column_stack([prediction[:, :5], actions[:, k + 1], np.repeat(future[None], count, axis=0)])
                    windows = np.concatenate([windows[:, 1:], nextrow[:, None]], axis=1)
            valid &= np.isfinite(costs)
            if time.perf_counter() - start >= cfg['failure_policy']['timeouts']['planning_seconds']:
                reason = 'timeout'
            if not reason and not valid.any():
                reason = 'no_valid_candidates'
        except Exception as exc:
            reason, error = 'planning_error', repr(exc)
        detail = dict(fallback_reason=reason, error=error, decision_time_s=time.perf_counter() - start,
                      valid_candidates=int(valid.sum()))
        if reason:
            return None, detail
        costs[~valid] = np.inf
        index = int(costs.argmin())
        detail['predicted_plan_cost'] = float(costs[index])
        return [float(actions[index, 0])], detail


def run_session(config, episode, recorded, model, name, condition, emit, progress, env_factory=ConfiguredBoptestEnv):
    """同一时间/预热/上下文启动真实回合；model=None 为内置控制，name=replay 表示全程重放。"""
    envcfg = config['env']
    context = envcfg['context_steps']
    steps = context + envcfg['episode_steps']
    local = dict(url=envcfg['url'], case=envcfg['case'], step_seconds=envcfg['step_seconds'],
                 warmup_seconds=envcfg['warmup_seconds'], start_time=episode['context_start_s'],
                 episode_steps=steps, measurements=['reaTZon_y'], failure_policy=config['failure_policy'],
                 action=dict(point=envcfg['action']['point'], activation=envcfg['action']['activation'],
                             experiment_bounds_c=envcfg['action']['bounds']))
    env = env_factory(local)
    controller = None if model is None else ConfiguredMPC(model, config)
    history, rows, transfers = [], [], []
    rng = np.random.default_rng(config['seed']['observation_fault']['base'] + episode['start_day'])
    previous_supply = None
    failure = None
    try:
        progress(phase='reset', model=name, episode=episode['episode_id'], condition=condition, step=-1)
        version = env.request('GET', '/version')
        if version.get('version') != envcfg['version']:
            raise ValueError(f'Service version mismatch: {version}')
        _, initial = env.reset(seed=config['seed']['candidate']['base'])
        progress(phase='initialized', model=name, episode=episode['episode_id'], condition=condition, step=-1, testid=env.testid)
        points = initial['measurements']
        previous = float(recorded.action_applied_c.iloc[0])
        forecast = env.request('PUT', f'/forecast/{env.testid}', dict(point_names=['LowerSetp[1]', 'UpperSetp[1]'],
                           horizon=(steps + config['horizon']['mpc_steps']) * 900, interval=900))
        context_kpi = None
        for k in range(steps):
            progress(phase='advance', model=name, episode=episode['episode_id'], condition=condition, step=k)
            timestamp = local['start_time'] + k * 900
            true = measurement_state(points)
            raw, observed, missing, previous_supply = corrupt_observation(true, condition, config['evaluator']['faults'],
                                                                         rng, previous_supply, k - context)
            weather = np.array([points['weaSta_reaWeaTDryBul_y'] - 273.15, points['weaSta_reaWeaHGloHor_y']])
            history.append(np.r_[observed, previous, exogenous_features(weather, np.asarray(timestamp))])
            history = history[-config['model']['history_steps']:]
            detail = dict(fallback_reason='', decision_time_s=0., error='', valid_candidates=None)
            shift = config['evaluator']['faults'].get(condition, {}).get('shift_degC', 0.)
            future = timestamp + np.arange(1, config['horizon']['mpc_steps'] + 1) * 900
            lower = np.interp(future, forecast['time'], forecast['LowerSetp[1]']) - 273.15 + shift
            upper = np.interp(future, forecast['time'], forecast['UpperSetp[1]']) - 273.15 + shift
            if k < context or name == 'replay':
                action = [float(recorded.action_applied_c.iloc[k])]
            elif controller is None:
                action = None
            else:
                action, detail = controller.act(np.asarray(history), timestamp, lower, upper, k, previous)
            _, _, terminated, truncated, info = (env.safe_step(action, detail['fallback_reason'])
                if controller is not None and k >= context else env.step(action))
            after = measurement_state(info['measurements'])
            applied = info['applied_action_c']
            history[-1][5] = applied
            violation = max(lower[0] - after[0], 0) + max(after[0] - upper[0], 0)
            weights = config['metrics']['cost_weights']
            cost = float(stage_cost(after[0], info['energy_kwh'], lower[0], upper[0], previous, applied,
                comfort_weight=weights['comfort'], energy_weight=weights['energy'], change_weight=weights['action_change']))
            row = dict(step=k, timestamp=timestamp, context=k < context, true_before=true.tolist(), raw_observation=raw.tolist(),
                observed_before=observed.tolist(), missing=missing.tolist(), true_after=after.tolist(), weather=weather.tolist(),
                requested_action_c=None if action is None else action[0], applied_action_c=applied,
                override_active=info['override_active'], lower_c=float(lower[0]), upper_c=float(upper[0]), zone_c=float(after[0]),
                energy_kwh=info['energy_kwh'], comfort_violation_c=violation, comfort_violation=violation > 0,
                tracking_error_c=abs(after[0] - (lower[0] + upper[0]) / 2), cost=cost,
                decision_time_s=detail['decision_time_s'], fallback_reason=info['fallback_reason'],
                planning_error=detail['error'], valid_candidates=detail.get('valid_candidates'), native_kpi=info['native_kpi'],
                terminated=terminated, truncated=truncated)
            emit(row)
            rows.append(row)
            transfers.append(dict(episode_id=episode['episode_id'], source_step=k, timestamp=timestamp,
                                  state=json.dumps(true.tolist()), next_state=json.dumps(after.tolist()),
                                  exogenous=json.dumps(weather.tolist()), action_applied_c=applied, energy_kwh=info['energy_kwh']))
            if k == context - 1:
                context_kpi = info['native_kpi'].copy()
            if terminated or truncated:
                if k != steps - 1 or terminated:
                    raise RuntimeError('Unexpected early simulator termination')
            previous, points = applied, info['measurements']
        if not rows[-1]['truncated']:
            raise RuntimeError('Episode did not reach time limit')
        scored = pd.DataFrame(rows[context:])
        metrics = calculate_boptest_control_metrics(scored)
        metrics.update(native_context_kpi=context_kpi, native_final_kpi=rows[-1]['native_kpi'],
                       native_interval_note='Only additive cumulative KPI differences are meaningful; do not subtract maxima or ratios.')
        return rows, metrics, pd.DataFrame(transfers)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            env.close()
        except Exception:
            if failure is None:
                raise


def frame_arrays(frame):
    """沿用 Week6 的 5观测+当前动作+6外生字段；目标含下一观测与本步原生能耗。"""
    states = np.stack(frame.state.map(json.loads))
    weather = np.stack(frame.exogenous.map(json.loads))
    features = np.column_stack([states, frame.action_applied_c.to_numpy(), exogenous_features(weather, frame.timestamp.to_numpy())])
    targets = np.column_stack([np.stack(frame.next_state.map(json.loads)), frame.energy_kwh.to_numpy()])
    return features, targets


def predict_window(model, features, start, horizon, config, condition, mode):
    """自由递推仅使用自身状态；TF 才读取后续观测，天气信息集由配置控制。保留失败步原值。"""
    length = model.length
    history = features[start - length + 1:start + 1].copy()
    rng = np.random.default_rng(config['seed']['observation_fault']['base'] + start)
    previous = None
    faults = config['evaluator']['faults']
    fault = faults.get(condition, {})
    for i in range(length):
        # 噪声污染完整初始历史；缺失在起点才开始，与 Week6 窗口语义保持一致。
        active = 0 if fault.get('kind') == 'temperature_noise' or i == length - 1 else -1
        _, history[i, :5], _, previous = corrupt_observation(history[i, :5], condition, faults, rng, previous, active)
    weather = features[start, 6:8].copy()
    raw = np.full((horizon, 6), np.nan)
    valid = np.zeros(horizon, bool)
    warning = np.zeros(horizon, bool)
    failure = None
    for k in range(horizon):
        try:
            value = np.asarray(model.predict(history), float)
            raw[k] = value
            threshold = config['metrics']['validity']
            if np.max(np.abs(value)) > threshold['absolute_divergence']:
                failure = dict(step=k + 1, reason='divergence')
                break
            if not np.isfinite(value).all():
                failure = dict(step=k + 1, reason='nonfinite')
                break
            valid[k] = True
            zone, water, energy = threshold['zone_degC'], threshold['water_degC'], threshold['interval_energy_kwh']
            warning[k] = not (zone[0] <= value[0] <= zone[1] and np.all((value[1:3] >= water[0]) & (value[1:3] <= water[1]))
                                  and value[4] >= 0 and energy[0] <= value[5] <= energy[1])
        except Exception as exc:
            if isinstance(exc, PredictionOutputError) and exc.prediction.shape == (6,):
                raw[k] = exc.prediction
            failure = dict(step=k + 1, reason=type(exc).__name__, message=str(exc))
            break
        if k + 1 < horizon:
            nextrow = features[start + k + 1].copy()
            if mode == 'free_running':
                nextrow[:5] = value[:5]
            else:
                _, nextrow[:5], _, previous = corrupt_observation(nextrow[:5], condition, faults, rng, previous, k + 1)
            if config['dataset']['weather_information'] == 'persistence_at_origin':
                nextrow[6:8] = weather
            history = np.vstack([history[1:], nextrow])
    return raw, valid, warning, failure


def endpoint_metrics(actual, predicted, valid):
    """无效请求保留在分母；误差仅用有效预测，变量单位分别报告。"""
    count = len(valid)
    if valid.any():
        mae, rmse = error_by_variable(actual[valid], predicted[valid])
    else:
        mae, rmse = np.full(6, np.nan), np.full(6, np.nan)
    return dict(requested=count, valid=int(valid.sum()), failed=int((~valid).sum()),
                valid_fraction=float(valid.mean()), mae=mae.tolist(), rmse=rmse.tolist())
