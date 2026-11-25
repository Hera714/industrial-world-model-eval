"""在完整 episode 内选共同起点，比较真实状态输入与模型自由递推。"""
import json
import numpy as np


def rollout_episode(model, x, u, d, y, horizon, mode, condition, divergence_bounds=(-100, 100), validity_config=None, physics=None):
    """批量独立起点；自由递推不读取未来状态，失败后保留前缀并停止该起点。"""
    if mode not in ('teacher_forcing', 'free_running') or condition not in ('recorded_observation', 'persistence_forecast'):
        raise ValueError('Unknown rollout mode or disturbance condition')
    x, u, d, y = [np.asarray(value, dtype=float) for value in (x, u, d, y)]
    if x.ndim != 2 or x.shape != y.shape or u.shape[0] != len(x) or d.shape[0] != len(x):
        raise ValueError('Invalid episode arrays')
    if not all(np.isfinite(value).all() for value in (x, u, d, y)) or not 1 <= horizon <= len(x):
        raise ValueError('Invalid episode values or horizon')
    starts = np.arange(len(x)-horizon+1)
    prediction = np.full((len(starts), horizon, x.shape[1]), np.nan)
    actual = np.stack([y[starts+k] for k in range(horizon)], axis=1)
    state = x[starts].copy()
    active = np.ones(len(starts), dtype=bool)
    failure_step = np.zeros(len(starts), dtype=int)
    reasons = np.full(len(starts), '', dtype=object)
    from metrics.validity import check_prediction
    validity_config = validity_config or dict(divergence_bounds_c=divergence_bounds)
    raw = np.full_like(prediction, np.nan)
    attempted = np.zeros((len(starts), horizon), dtype=bool)
    events = []
    for k in range(horizon):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        indices = starts[ids]+k
        # 未来天气观测仅在离线给定扰动条件中可见；保持预测只读取起点室外观测。
        disturbance = d[indices].copy()
        if condition == 'persistence_forecast':
            disturbance[:, 0] = d[starts[ids], 0]
        inputs = x[indices] if mode == 'teacher_forcing' else state[ids]
        output = np.full_like(inputs, np.nan)
        errors = ['']*len(ids)
        try:
            model.reset()
            output = np.asarray(model.predict_next(inputs, u[indices], disturbance), dtype=float)
            if output.shape != inputs.shape or not np.isfinite(output).all():
                raise ValueError('Invalid batch prediction')
        except Exception:
            # 批量中某个起点失败不能连带丢弃其他起点；逐个重试并记录实际异常。
            output = np.full_like(inputs, np.nan)
            for j in range(len(ids)):
                try:
                    model.reset()
                    value = np.asarray(model.predict_next(inputs[j], u[indices[j]], disturbance[j]), dtype=float)
                    if value.shape != inputs[j].shape:
                        raise ValueError('Invalid prediction')
                    output[j] = value
                except Exception as exc:
                    errors[j] = f'{type(exc).__name__}: {exc}'
                    rejected = getattr(exc, 'prediction', None)
                    if rejected is not None and np.asarray(rejected).shape == inputs[j].shape:
                        output[j] = rejected
                        if not np.isfinite(rejected).all():
                            errors[j] = ''
        diagnostics = check_prediction(inputs, output, validity_config, physics)
        for j, origin in enumerate(ids):
            attempted[origin, k] = True
            raw[origin, k] = output[j]
            if errors[j]:
                active[origin] = False
                failure_step[origin] = k+1
                reasons[origin] = errors[j]
                events.append(dict(origin=int(origin), horizon=k+1, variable='all', event='inference_error',
                                   previous=None, predicted=None, magnitude=None, low=None, high=None,
                                   detail=errors[j], stops=True))
                continue
            for event, mask in diagnostics['events'].items():
                for index in np.flatnonzero(mask[j]):
                    detail = diagnostics['details'].get(event)
                    events.append(dict(origin=int(origin), horizon=k+1, variable=('air', 'wall')[index],
                        event=event, previous=float(inputs[j, index]), predicted=float(output[j, index]),
                        magnitude=float(detail['magnitude'][j, index]) if detail else None,
                        low=float(detail['low'][j, index]) if detail else None,
                        high=float(detail['high'][j, index]) if detail else None, detail='',
                        stops=event in ('nan', 'inf', 'divergence')))
            if not np.isfinite(output[j]).all():
                active[origin] = False
                failure_step[origin] = k+1
                reasons[origin] = 'nonfinite'
                continue
            prediction[origin, k] = output[j]
            state[origin] = output[j]
            if diagnostics['stop'][j]:
                active[origin] = False
                failure_step[origin] = k+1
                reasons[origin] = 'divergence_bounds'
    return dict(predicted=prediction, actual=actual, failure_step=failure_step, failure_reason=reasons,
                raw=raw, attempted=attempted, events=events)


def rollout_errors(actual, predicted, horizon):
    """分别计算末端、完整前缀和可用前缀；同时返回全部请求起点的有效计数。"""
    from metrics.prediction import mae, rmse
    a, p = actual[:, :horizon], predicted[:, :horizon]
    valid = np.isfinite(p).all(axis=-1)
    for definition in ('terminal', 'prefix_complete', 'prefix_available'):
        if definition == 'terminal':
            selected = valid[:, -1]
            aa, pp = a[selected, -1], p[selected, -1]
        elif definition == 'prefix_complete':
            selected = valid.all(axis=1)
            aa, pp = a[selected].reshape(-1, a.shape[-1]), p[selected].reshape(-1, p.shape[-1])
        else:
            selected = valid.any(axis=1)
            aa, pp = a[valid], p[valid]
        variables = ('air', 'wall') if a.shape[-1] == 2 else ('setpoint','velocity','gain','shift','fatigue','consumption')
        for index, variable in enumerate(variables):
            yield dict(definition=definition, variable=variable, requested=len(a),
                valid_origins=int(selected.sum()), complete_origins=int(valid.all(axis=1).sum()),
                valid_points=len(aa), valid_fraction=float(selected.mean()),
                mae_c=mae(aa[:, index:index+1], pp[:, index:index+1]) if len(aa) else np.nan,
                rmse_c=rmse(aa[:, index:index+1], pp[:, index:index+1]) if len(aa) else np.nan)


def summarize_rollout_metrics(metrics):
    """分别汇总 episode 和训练种子；失败数与误差可用数保持独立定义。"""
    from evaluators.summary import summarize_episode_metrics
    return summarize_episode_metrics(metrics)


def evaluate_week5_rollouts(config_path, root='.', device='cpu'):
    """冻结热工主实验：两种模式×两种天气信息集，输出逐步及关键 horizon 指标。"""
    from pathlib import Path
    import hashlib
    import pandas as pd
    import yaml
    from models.dynamics_base import load_dynamics
    root = Path(root).resolve()
    config = yaml.safe_load(Path(config_path).read_text())
    if config['environment'] != 'simple_thermal':
        raise ValueError('Frozen thermal evaluator only')
    for relative, expected in config['sha256'].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen file changed: {relative}')
    splits = config['data']['split_ids']
    if any(set(splits[a]) & set(splits[b]) for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Overlapping episode splits')
    frame = pd.read_csv(root/config['data']['directory']/'test.csv', skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    frame['episode_id'] = frame.episode_id.str.strip()
    if set(frame.episode_id) != set(splits['test']) or len(splits['test']) != config['origins']['expected_test_episodes']:
        raise ValueError('Frozen episodes mismatch')
    episodes = []; origins = []
    for episode_id in splits['test']:
        episode = frame[frame.episode_id == episode_id].sort_values('source_step')
        if not np.array_equal(episode.source_step, np.arange(len(episode))):
            raise ValueError('Noncontiguous episode')
        arrays = [np.stack(episode[key].map(json.loads)) for key in
                  ('state', 'action_applied', 'exogenous_available_at_decision', 'next_state')]
        count = len(episode)-config['max_horizon']+1
        if count != config['origins']['expected_origins_per_episode']:
            raise ValueError('Frozen origin count mismatch')
        episodes.append((episode_id, arrays, json.loads(episode.iloc[0].environment_config)))
        origins.extend(dict(episode_id=episode_id, source_step=k, policy=episode.iloc[0].policy,
                            initial_air_c=float(arrays[0][0, 0])) for k in range(count))
    output = root/config['output']/'rollout'
    output.mkdir(parents=True, exist_ok=True)
    metric_rows = []; failures = []; validity_events = []; validity_origins = []
    for spec in config['models']:
        model = load_dynamics(spec, root, device, hashes=config['sha256'])
        family = spec.get('family') or ('residual_mlp' if spec['name'].startswith('residual_') else spec['kind'])
        saved = {}
        for condition in config['information']['conditions']:
            for mode in config['modes']:
                accumulated = []; raw_outputs = []; attempted_outputs = []
                for episode_id, arrays, physics in episodes:
                    result = rollout_episode(model, *arrays, config['max_horizon'], mode, condition,
                                             config['validity']['divergence_bounds_c'], config['validity'], physics)
                    accumulated.append(result['predicted'])
                    raw_outputs.append(result['raw']); attempted_outputs.append(result['attempted'])
                    for event in result['events']:
                        validity_events.append(dict(model=spec['name'], mode=mode, condition=condition,
                            episode_id=episode_id, source_step=event['origin'], **event))
                    for k, (step, reason) in enumerate(zip(result['failure_step'], result['failure_reason'])):
                        failures.append(dict(model=spec['name'], mode=mode, condition=condition,
                            episode_id=episode_id, source_step=k, failure_step=int(step), reason=reason))
                        own = [event for event in result['events'] if event['origin'] == k]
                        counts = {name+'_steps': len({event['horizon'] for event in own if event['event'] == name})
                            for name in ('nan', 'inf', 'inference_error', 'observation_bounds', 'plausibility_bounds', 'rate', 'divergence')}
                        validity_origins.append(dict(model=spec['name'], mode=mode, condition=condition,
                            episode_id=episode_id, source_step=k, requested_steps=config['max_horizon'],
                            attempted_steps=int(result['attempted'][k].sum()),
                            finite_steps=int(np.isfinite(result['predicted'][k]).all(axis=-1).sum()),
                            first_anomaly_step=min((event['horizon'] for event in own), default=0),
                            first_failure_step=int(step), failure_reason=reason, **counts))
                    for horizon in range(1, config['max_horizon']+1):
                        for metric in rollout_errors(result['actual'], result['predicted'], horizon):
                            metric_rows.append(dict(model=spec['name'], family=family, seed=spec.get('seed', -1),
                                mode=mode, condition=condition, episode_id=episode_id, horizon=horizon,
                                failed_origins=int(((result['failure_step'] > 0) & (result['failure_step'] <= horizon)).sum()), **metric))
                saved[condition+'__'+mode] = np.concatenate(accumulated)
                saved[condition+'__'+mode+'__raw'] = np.concatenate(raw_outputs)
                saved[condition+'__'+mode+'__attempted'] = np.concatenate(attempted_outputs)
        saved['actual'] = np.concatenate([np.stack([arrays[3][np.arange(len(arrays[0])-config['max_horizon']+1)+k]
            for k in range(config['max_horizon'])], axis=1) for _, arrays, _ in episodes])
        np.savez_compressed(output/(spec['name']+'_predictions.npz'), **saved)
        print(f"Completed {spec['name']}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    summary, seeds = summarize_rollout_metrics(metrics)
    metrics.to_csv(output/'metrics_by_episode.csv', index=False)
    summary.to_csv(output/'metrics_by_horizon.csv', index=False)
    summary[summary.horizon.isin(config['horizons'])].to_csv(output/'metrics_key_horizons.csv', index=False)
    seeds.to_csv(output/'metrics_across_seeds.csv', index=False)
    pd.DataFrame(origins).to_csv(output/'origins.csv', index=False)
    pd.DataFrame(failures).to_csv(output/'failures.csv', index=False)
    event_columns = ['model','mode','condition','episode_id','source_step','origin','horizon','variable',
                     'event','previous','predicted','magnitude','low','high','detail','stops']
    pd.DataFrame(validity_events, columns=event_columns).to_csv(output/'validity_events.csv.gz', index=False, compression='gzip')
    validity = pd.DataFrame(validity_origins)
    validity.to_csv(output/'validity_by_origin.csv', index=False)
    validity['has_failure'] = validity.first_failure_step > 0
    validity['has_anomaly'] = validity.first_anomaly_step > 0
    totals = validity.groupby(['model', 'mode', 'condition']).agg(
        requested_origins=('source_step','size'), failed_origins=('has_failure','sum'),
        flagged_origins=('has_anomaly','sum'), requested_steps=('requested_steps','sum'),
        attempted_steps=('attempted_steps','sum'), finite_steps=('finite_steps','sum'))
    for name in ('nan','inf','inference_error','observation_bounds','plausibility_bounds','rate','divergence'):
        totals[name+'_steps'] = validity.groupby(['model','mode','condition'])[name+'_steps'].sum()
    totals['failure_fraction'] = totals.failed_origins/totals.requested_origins
    totals['finite_fraction'] = totals.finite_steps/totals.requested_steps
    totals['observation_bounds_applicable'] = any(value is not None for value in config['validity']['observation_bounds_c'])
    totals.to_csv(output/'validity_summary.csv')
    snapshot = dict(config=config, device=device, units='Celsius',
        array_axes=['origin (see origins.csv)', 'horizon 1..80', 'air / wall'],
        model_manifest=config['models'],
        prefix='MAE and sqrt(mean squared error) over steps 1..H, then episode equal weighting',
        divergence='Keep finite threshold-crossing prediction; stop subsequent steps',
        future_inputs='Recorded applied actions; known load; outdoor according to separately named condition')
    (output/'protocol_snapshot.json').write_text(json.dumps(snapshot, indent=2))
    print(f'Saved rollout results: {output}', flush=True)
    from scripts.summarize_week5 import summarize_folder
    summarize_folder(output)
    return summary


def evaluate_rollouts(model, frame, horizon=20):
    """动作采用记录序列，扰动采用各步当前可见值；返回可追溯的三维预测。"""
    predictions, teachers, truths, origins = [], [], [], []
    for episode_id, episode in frame.groupby('episode_id', sort=True):
        episode = episode.sort_values('source_step')
        if np.any(np.diff(episode.source_step.to_numpy()) != 1):
            raise ValueError('Non-contiguous episode')
        x = np.stack(episode.state.map(json.loads))
        u = np.stack(episode.action_applied.map(json.loads))
        d = np.stack(episode.exogenous_available_at_decision.map(json.loads))
        y = np.stack(episode.next_state.map(json.loads))
        # 全部 horizon 使用能支持完整 20 步的同一批起点。
        starts = np.arange(len(episode)-horizon+1)
        if not len(starts):
            raise ValueError('Episode shorter than rollout')
        predicted = x[starts].copy()
        free, teacher = [], []
        for step in range(horizon):
            indices = starts+step
            predicted = model.predict_next(predicted, u[indices], d[indices])
            free.append(predicted.copy())
            teacher.append(model.predict_next(x[indices],u[indices],d[indices]))
        predictions.append(np.stack(free,axis=1))
        teachers.append(np.stack(teacher,axis=1))
        truths.append(np.stack([y[starts+k] for k in range(horizon)],axis=1))
        origins.extend((episode_id,int(episode.iloc[start].source_step)) for start in starts)
    return dict(free=np.concatenate(predictions), teacher=np.concatenate(teachers),
                actual=np.concatenate(truths), origins=origins)
