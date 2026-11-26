"""总说明：一个配置完成预检、模型加载、BOPTEST 采集/控制、开环/排序评测及独立 run 报告。"""
from pathlib import Path
import json
import hashlib
import yaml
import numpy as np
import pandas as pd
import sys
from contextlib import contextmanager
from benchmark.config import load_config, validate_config
from benchmark.loaders import load_models
from benchmark.artifacts import RunArtifacts, atomic_write
from evaluators.benchmark_boptest import (ConfiguredBoptestEnv, run_session, frame_arrays,
                                          predict_window, endpoint_metrics)
from evaluators.consistency import score_fixed_action_plans
from controllers.boptest_mpc import stage_cost
from models.boptest_dynamics import exogenous_features
from evaluators.summary import summarize_episode_metrics

VARIABLES = ['zone_c', 'supply_c', 'return_c', 'floor_heat_kw', 'electric_kw', 'interval_energy_kwh']
UNITS = ['degC', 'degC', 'degC', 'kW', 'kW', 'kWh']


@contextmanager
def capture_console(run):
    """同时保留控制台与第三方输出；异常堆栈另由 failure.json 保存。"""
    class Tee:
        def __init__(self, stream):
            self.stream = stream
        def write(self, message):
            self.stream.write(message)
            with (run.path/'run.log').open('a', encoding='utf-8') as log:
                log.write(message)
            return len(message)
        def flush(self):
            self.stream.flush()
        def __getattr__(self, key):
            return getattr(self.stream, key)
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = Tee(old_out), Tee(old_err)
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _prediction_summaries(records):
    """适配已有 episode→训练种子汇总，不把重叠窗口当作独立重复。"""
    if not records:
        return [], []
    expanded = []
    for row in records:
        for index, (variable, unit) in enumerate(zip(VARIABLES, UNITS)):
            expanded.append(dict(model=row['model'], family=row['family'], seed=row['training_seed'] if row['training_seed'] is not None else -1,
                mode=row['mode'], condition=row['condition'], horizon=row['horizon'], definition='endpoint', variable=variable,
                unit=unit, episode_id=row['episode_id'], mae_native=row['mae'][index], rmse_native=row['rmse'][index],
                requested=row['requested'], valid_origins=row['valid'], complete_origins=row['valid'],
                valid_points=row['valid'], failed_origins=row['failed']))
    by_model, by_seed = summarize_episode_metrics(pd.DataFrame(expanded))
    return by_model.to_dict('records'), by_seed.to_dict('records')


def _record(run, task, model, episode, condition, step, payload):
    """所有阶段共享 CSV 外层列，详细观测、动作、约束与模型预测放在 payload 中。"""
    run.append_trajectory(dict(task=task, model=model, episode_id=episode, condition=condition,
                              step=step, payload=payload))


def _open_loop(run, cfg, models, entries, frame_for, raw_arrays):
    """同一起点/动作计划评测全部模型，按 episode 汇总，失败窗口仍计入 requested。"""
    records = []
    matrix = cfg['evaluator']['condition_matrix']['open_loop']
    horizons = cfg['horizon']['prediction_steps']
    for eid in matrix['episodes']:
        features, targets = frame_arrays(frame_for(eid))
        context = cfg['env']['context_steps']
        origins = list(range(context, context + cfg['env']['episode_steps'] - max(horizons) + 1,
                             cfg['evaluator']['origin_stride_steps']))
        if not origins:
            raise ValueError('No complete open-loop windows: ' + eid)
        for name, model in models.items():
            for condition in matrix['conditions']:
                for mode in cfg['evaluator']['prediction_modes']:
                    key = f'open__{eid}__{name}__{condition}__{mode}'
                    predictions = np.full((len(origins), max(horizons), 6), np.nan)
                    valid = np.zeros((len(origins), max(horizons)), bool)
                    warnings = valid.copy()
                    truth = np.stack([targets[start:start + max(horizons)] for start in origins])
                    for index, start in enumerate(origins):
                        run.set_context(task='open_loop', model=name, episode=eid, condition=condition, mode=mode, origin=start)
                        model.reset()
                        predicted, good, warning, failure = predict_window(model, features, start, max(horizons), cfg, condition, mode)
                        predictions[index], valid[index], warnings[index] = predicted, good, warning
                        if failure:
                            run.log('prediction_failed', level='WARNING', failure=failure)
                        for k in range(max(horizons)):
                            _record(run, 'open_loop', name, eid, condition, start + k, dict(mode=mode, origin=start,
                                horizon=k + 1, actual=truth[index, k], prediction=predicted[k], valid=bool(good[k]),
                                physical_warning=bool(warning[k]), failure=failure))
                    raw_arrays[key+'__prediction'] = predictions
                    raw_arrays[key+'__actual'] = truth
                    raw_arrays[key+'__valid'] = valid
                    raw_arrays[key+'__origins'] = np.asarray(origins)
                    for horizon in horizons:
                        metric = endpoint_metrics(truth[:, horizon - 1], predictions[:, horizon - 1], valid[:, horizon - 1])
                        records.append(dict(model=name, family=model.kind, training_seed=cfg['seed']['model'][name],
                            episode_id=eid, split=entries[eid]['split'], condition=condition, mode=mode, horizon=horizon,
                            physical_warning_fraction=float(warnings[:, horizon - 1].mean()), **metric))
        run.log('open_loop_episode_completed', episode=eid, models=len(models), origins=len(origins))
        print(f'open_loop completed: {eid} ({len(models)} models)', flush=True)
    return records


def _ranking(run, cfg, models, sessions, arrays):
    """把当前 run 的 clean 闭环实际动作冻结，复用排序指标；并非反馈策略的反事实评测。"""
    records = []
    matrix = cfg['evaluator']['condition_matrix']['policy_ranking']
    horizon = max(cfg['horizon']['prediction_steps'])
    context = cfg['env']['context_steps']
    for eid in matrix['episodes']:
        for condition in matrix['conditions']:
            plans = {name: rows for (task, episode, name, cond), rows in sessions.items()
                     if task == 'closed_loop' and episode == eid and cond == condition}
            if len(plans) < 2:
                raise ValueError(f'Ranking requires at least two matching closed-loop plans: {eid}/{condition}')
            first = next(iter(plans.values()))
            # 同一已发生的初始历史才可比较动作计划，禁止每条计划使用各自漂移后的起点。
            for rows in plans.values():
                np.testing.assert_allclose([r['true_before'] for r in rows[:context + 1]],
                                           [r['true_before'] for r in first[:context + 1]], atol=1e-5)
            for name, model in models.items():
                costs = []
                for policy, rows in plans.items():
                    features = np.stack([np.r_[r['true_before'], r['applied_action_c'],
                        # 时间周期必须来自实际时间，而非数组下标。
                        exogenous_features(
                            np.asarray(r['weather']), np.asarray(r['timestamp']))] for r in rows])
                    predicted, valid, _, failure = predict_window(model, features, context, horizon, cfg, 'clean', 'free_running')
                    good = bool(valid.all() and np.all(predicted[:, 5] >= 0))
                    previous = rows[context - 1]['applied_action_c']
                    value = 0.
                    weights = cfg['metrics']['cost_weights']
                    for k, row in enumerate(rows[context:context + horizon]):
                        if good:
                            value += float(stage_cost(predicted[k, 0], predicted[k, 5], row['lower_c'], row['upper_c'],
                                previous, row['applied_action_c'], comfort_weight=weights['comfort'],
                                energy_weight=weights['energy'], change_weight=weights['action_change']))
                        previous = row['applied_action_c']
                    entry = dict(policy=policy, true_cost=sum(r['cost'] for r in rows[context:context + horizon]),
                                 predicted_cost=value if good else np.nan, valid=good, failure=failure)
                    costs.append(entry)
                    _record(run, 'policy_ranking', name, eid, condition, context,
                            dict(horizon=horizon, **entry))
                    arrays[f'ranking__{eid}__{condition}__{name}__{policy}'] = predicted
                records.append(dict(episode_id=eid, model=name, condition=condition, horizon=horizon,
                                    plans=costs, **score_fixed_action_plans(costs)))
    return records


def _report(metrics):
    """逐回合表保留指标来源，避免把 °C 与 kWh 的误差混成一个总 RMSE。"""
    lines = ['# Benchmark run', '', f"Run: `{metrics['run_id']}`", '',
             '所有闭环动作均由 BOPTEST 服务执行；开环输入模式与天气信息集见 config.yaml。', '',
             '## 开环指标（每个 episode）', '',
             '| 模型 | Episode | 条件 | 模式 | Horizon | 室温 RMSE °C | 有效比例 |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in metrics['open_loop']:
        lines.append(f"| {row['model']} | {row['episode_id']} | {row['condition']} | {row['mode']} | {row['horizon']} | {row['rmse'][0]:.6g} | {row['valid_fraction']:.3f} |")
    lines += ['', '完整六变量 MAE/RMSE、失败数与物理异常比例保存在 metrics.json。', '',
              '## 真实闭环指标', '', '| 任务 | 模型 | Episode | 条件 | 成本 | 能耗 kWh | fallback |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for row in metrics['closed_loop']:
        lines.append(f"| {row['task']} | {row['model']} | {row['episode_id']} | {row['condition']} | {row['cost']:.6g} | {row['energy_kwh']:.6g} | {row['fallback_count']} |")
    lines += ['', '原生 KPI 的 final 区间包含历史上下文；只对可加的累积量作差，不对峰值或比率作差。', '',
              f"排序评测条目：{len(metrics['policy_ranking'])}。各回合独立保留，未将时间步视为独立重复。", '',
              'status.json 的 completed 表示流程完成；有效比例与失败事件需同时查看。', '']
    return '\n'.join(lines)


def run_benchmark(config, root=None, env_factory=ConfiguredBoptestEnv, resume_from=None, tracking_config=None):
    """入口只编排，真实回合、动力学预测和指标分别复用各模块；异常交给产物上下文保存。"""
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    cfg = load_config(config, root) if isinstance(config, (str, Path)) else validate_config(config, root)
    # 跨任务依赖与回放动作先检查，不能等服务启动后才发现配置无法执行。
    ledger = json.loads((root/cfg['dataset']['manifest']['path']).read_text())
    entries = {e['episode_id']: e for e in ledger['episodes']}
    for eid in cfg['dataset']['evaluation_episodes']:
        if entries[eid]['context_steps'] != cfg['env']['context_steps']:
            raise ValueError('env.context_steps must match frozen target boundary: '+eid)
    matrix = cfg['evaluator']['condition_matrix']
    if 'policy_ranking' in matrix:
        if len(cfg['model']['instances']) + int(cfg['evaluator']['baseline'] == 'embedded') < 2:
            raise ValueError('Ranking requires at least two executed policies')
        if (not set(matrix['policy_ranking']['episodes']) <= set(matrix['closed_loop']['episodes'])
                or matrix['policy_ranking']['conditions'] != ['clean']
                or 'clean' not in matrix['closed_loop']['conditions']):
            raise ValueError('Ranking requires matching clean closed-loop episodes')
    for eid in cfg['dataset']['evaluation_episodes']:
        frame = pd.read_csv(root/entries[eid]['file']['path'])
        count = cfg['env']['context_steps'] + (cfg['env']['episode_steps'] if cfg['dataset']['mode'] == 'online' else 0)
        actions = frame.action_applied_c.to_numpy()[:count]
        lo, hi = cfg['env']['action']['bounds']
        if not np.isfinite(actions).all() or np.any((actions < lo) | (actions > hi)):
            raise ValueError('Recorded replay actions exceed configured bounds: '+eid)
    # 续跑前校验完整配置和全部输入哈希；旧 run 保持只读，避免覆盖失败上下文。
    source = Path(resume_from).resolve() if resume_from else None
    if source:
        if yaml.safe_load((source/'config.yaml').read_text()) != cfg:
            raise ValueError('Resume configuration mismatch')
        old_hashes = json.loads((source/'input_hashes.json').read_text())
        snapshot = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        if old_hashes['config_sha256'] != hashlib.sha256(snapshot.encode()).hexdigest():
            raise ValueError('Resume configuration hash mismatch')
        for relative, expected in old_hashes['files'].items():
            file = (root / relative).resolve()
            if not file.is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError('Resume input hash mismatch: '+relative)
    with RunArtifacts(cfg, root, tracking_config=tracking_config) as run, capture_console(run):
        print('Run directory: ' + str(run.path), flush=True)
        run.set_context(phase='load_models')
        models = load_models(cfg, root)
        raw_arrays, sessions, online_frames = {}, {}, {}
        metrics = dict(run_id=run.run_id, variables=VARIABLES, units=UNITS, open_loop=[], closed_loop=[], policy_ranking=[])

        def progress(**context):
            run.set_context(**context)
            if context.get('step', -1) % 24 == 0:
                run.log('episode_progress', **context)
                print(f"{context['model']} {context['episode']} {context['condition']} step={context['step']}", flush=True)

        def recorded(eid):
            frame = pd.read_csv(root/entries[eid]['file']['path'])
            if not np.array_equal(frame.source_step.to_numpy(), np.arange(len(frame))):
                raise ValueError('Nonconsecutive episode steps: '+eid)
            return frame

        def session(task, eid, name, condition, model):
            # 每个完整服务回合单独原子保存；不完整回合从头开始，禁止重复 advance。
            identity = dict(task=task, episode=eid, model=name, condition=condition)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            target = run.path/'episodes'/key
            old = source/'episodes'/key if source else None
            rows = values = transfer = None
            if old and (old/'complete.json').is_file():
                marker = json.loads((old/'complete.json').read_text())
                payload_bytes = (old/'episode.json').read_bytes()
                if marker['identity'] != identity or marker['sha256'] != hashlib.sha256(payload_bytes).hexdigest():
                    raise ValueError('Resume completed episode integrity mismatch: '+key)
                saved = json.loads(payload_bytes)
                rows, values = saved['rows'], saved['metrics']
                transfer = pd.DataFrame(saved['transfers'])
                expected = cfg['env']['context_steps'] + cfg['env']['episode_steps']
                if len(rows) != expected or not rows[-1]['truncated'] or rows[-1]['terminated']:
                    raise ValueError('Resume episode is incomplete: '+key)
                for row in rows: _record(run, task, name, eid, condition, row['step'], row)
                run.log('episode_reused', source=str(source), **identity)
                print(f'Reused {task} {eid} {name} {condition}', flush=True)
            else:
                run.set_context(task=task)
                rows, values, transfer = run_session(cfg, entries[eid], recorded(eid), model, name, condition,
                    emit=lambda row: _record(run, task, name, eid, condition, row['step'], row),
                    progress=progress, env_factory=env_factory)
            # 内容先落盘，完成标记最后写入。没有标记的目录永远不会复用。
            target.mkdir(parents=True, exist_ok=False)
            payload = json.dumps(dict(rows=rows, metrics=values, transfers=transfer.to_dict('records')), allow_nan=True, default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist())
            atomic_write(target/'episode.json', payload)
            atomic_write(target/'complete.json', json.dumps(dict(identity=identity, sha256=hashlib.sha256(payload.encode()).hexdigest())))
            sessions[task, eid, name, condition] = rows
            run.log('episode_completed', task=task, model=name, episode=eid, condition=condition, metrics=values)
            return values, transfer

        def frame_for(eid):
            if cfg['dataset']['mode'] == 'recorded':
                return recorded(eid)
            if eid not in online_frames:
                _, online_frames[eid] = session('collection', eid, 'replay', 'clean', None)
            return online_frames[eid]

        if 'open_loop' in cfg['evaluator']['tasks']:
            metrics['open_loop'] = _open_loop(run, cfg, models, entries, frame_for, raw_arrays)
            run.write_metrics(metrics)
        for task in ('closed_loop', 'robustness'):
            if task not in cfg['evaluator']['tasks']:
                continue
            matrix = cfg['evaluator']['condition_matrix'][task]
            for eid in matrix['episodes']:
                for condition in matrix['conditions']:
                    policies = list(models.items())
                    if cfg['evaluator']['baseline'] == 'embedded':
                        policies += [('embedded', None)]
                    for name, model in policies:
                        values, _ = session(task, eid, name, condition, model)
                        metrics['closed_loop'].append(dict(task=task, model=name, episode_id=eid,
                            family='embedded' if model is None else model.kind, condition=condition,
                            training_seed=cfg['seed']['model'].get(name), **values))
                        run.write_metrics(metrics)
        if 'policy_ranking' in cfg['evaluator']['tasks']:
            metrics['policy_ranking'] = _ranking(run, cfg, models, sessions, raw_arrays)
        metrics['prediction_episode_summary'], metrics['prediction_seed_summary'] = _prediction_summaries(metrics['open_loop'])
        if raw_arrays:
            run.write_predictions(**raw_arrays)
        run.write_metrics(metrics)
        run.write_report(_report(metrics))
    return run.path
