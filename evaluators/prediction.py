"""读取以摄氏度保存的预测，输出误差表、逐窗口轨迹和图。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics.prediction import mae, rmse, error_by_horizon


def evaluate(input_dir, output_dir, window_index=0):
    input_dir = Path(input_dir); output_dir = Path(output_dir)
    config = json.loads((input_dir / "config.json").read_text(encoding="utf-8"))
    with np.load(input_dir / "test_predictions.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    actual = arrays.pop("actual"); starts = arrays.pop("target_start")
    if actual.ndim != 2 or len(starts) != len(actual) or not 0 <= window_index < len(actual):
        raise ValueError("真实值、时间戳或窗口索引不合法")
    if config["horizon"] != actual.shape[1]:
        raise ValueError("预测长度与配置不一致")
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = []; overall = []; traces = []
    for name, prediction in arrays.items():
        errors_mae, errors_rmse = error_by_horizon(actual, prediction)
        overall.append({"model": name, "mae_c": mae(actual, prediction), "rmse_c": rmse(actual, prediction)})
        for index in range(actual.shape[1]):
            horizons.append({"model": name, "horizon": index+1, "minutes": (index+1)*10,
                             "mae_c": errors_mae[index], "rmse_c": errors_rmse[index]})
        # 每行保留起始时间、目标时间、真实值及预测值，便于复核。
        rows, cols = np.indices(actual.shape)
        target_times = starts[rows.ravel()] + cols.ravel().astype("timedelta64[m]") * 10
        traces.append(pd.DataFrame({"model": name, "window_index": rows.ravel(), "target_start": starts[rows.ravel()],
                                   "target_time": target_times, "horizon": cols.ravel()+1,
                                   "actual_c": actual.ravel(), "predicted_c": prediction.ravel()}))
    per_h = pd.DataFrame(horizons); totals = pd.DataFrame(overall)
    per_h.to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    per_h[per_h.horizon.isin([1, 6, 12, 24])].to_csv(output_dir / "metrics_key_horizons.csv", index=False)
    totals.to_csv(output_dir / "metrics_overall.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, group in per_h.groupby("model", sort=False):
        axes[0].plot(group.horizon, group.mae_c, label=name)
        axes[1].plot(group.horizon, group.rmse_c, label=name)
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        ax.set(xlabel="Forecast horizon (10-min steps)", ylabel=f"{metric} (°C)")
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "error_by_horizon.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    steps = np.arange(1, actual.shape[1] + 1)
    ax.plot(steps, actual[window_index], label="actual", color="black", linewidth=2)
    for name, prediction in arrays.items(): ax.plot(steps, prediction[window_index], label=name)
    ax.set(xlabel="Forecast horizon (10-min steps)", ylabel="T1 (°C)",
           title=f"Test window {window_index}, first target {starts[window_index]}")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "forecast_example.png", dpi=160); plt.close(fig)
    print(totals.to_string(index=False)); print(f"结果已保存至 {output_dir}")


def main():
    p = argparse.ArgumentParser(description="评测并绘制 Week 1 测试集预测")
    p.add_argument("--input", default="reports/week1_baselines")
    p.add_argument("--output", default="reports/week1_evaluation")
    p.add_argument("--window-index", type=int, default=0)
    args = p.parse_args(); evaluate(args.input, args.output, args.window_index)

def summarize_one_step(predictions):
    """先逐 episode 计算原单位误差，再按工况等权汇总；失败不从分母消失。"""
    episode_rows = []
    # 初温是回合起始真值，只用于分组；动作组允许一个回合出现在多个组中。
    groupings = {'all': [], 'policy': ['policy'], 'initial_air': ['initial_air_bin'],
                 'action': ['action_bin'], 'policy_initial_air': ['policy', 'initial_air_bin']}
    for grouping, fields in groupings.items():
        keys = ['model', 'family', 'seed', 'variable', 'episode_id'] + fields
        for key, group in predictions.groupby(keys, dropna=False, sort=True):
            row = dict(zip(keys, key))
            finite = np.isfinite(group.predicted.to_numpy()) & np.isfinite(group.actual.to_numpy())
            valid = group.loc[finite]
            row.update(grouping=grouping, group='all' if not fields else '|'.join(str(row[f]) for f in fields),
                       requested=len(group), valid=len(valid), failed=len(group)-len(valid),
                       valid_fraction=float(finite.mean()), unit='Celsius')
            row['mae_c'] = mae(valid[['actual']], valid[['predicted']]) if len(valid) else np.nan
            row['rmse_c'] = rmse(valid[['actual']], valid[['predicted']]) if len(valid) else np.nan
            episode_rows.append(row)
    episodes = pd.DataFrame(episode_rows)
    from evaluators.summary import summarize_episode_metrics
    standardized=episodes.rename(columns={'valid':'valid_origins','failed':'failed_origins'}).copy()
    standardized['complete_origins']=standardized.valid_origins
    standardized['valid_points']=standardized.valid_origins
    summary,seeds=summarize_episode_metrics(standardized,context=['variable','grouping','group'])
    summary=summary.rename(columns={'valid_origins':'valid','failed_origins':'failed'})
    return episodes,summary,seeds


def evaluate_one_step(config_path, root='.', device='cpu'):
    """按冻结配置评估热工模型；仅用当前真实状态、已生效动作和当前可得扰动。"""
    import hashlib
    import yaml
    from models.dynamics_base import load_dynamics
    root = Path(root).resolve()
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    if config['environment'] != 'simple_thermal':
        raise ValueError('This frozen evaluator supports simple_thermal only')
    # 在开始预测或写结果前核验所有冻结资产，发现变化直接停止。
    for relative, expected in config['sha256'].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen file changed: {relative}')
    splits = config['data']['split_ids']
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(splits[a]) & set(splits[b]):
            raise ValueError('Episode splits overlap')
    frame = pd.read_csv(root/config['data']['directory']/'test.csv', skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    for column in ('episode_id', 'scenario_id'):
        frame[column] = frame[column].str.strip()
    if set(frame.episode_id) != set(splits['test']):
        raise ValueError('Test episodes differ from frozen manifest')
    records = []
    for episode_id in splits['test']:
        episode = frame[frame.episode_id == episode_id].sort_values('source_step')
        if not np.array_equal(episode.source_step.to_numpy(), np.arange(len(episode))):
            raise ValueError('Episode steps must be unique and contiguous from zero')
        initial_air = float(json.loads(episode.iloc[0]['state'])[0])
        initial_bin = '<19' if initial_air < 19 else ('19-21' if initial_air < 21 else '>=21')
        count = len(episode)-config['max_horizon']+1
        if count != config['origins']['expected_origins_per_episode']:
            raise ValueError('Origin count differs from protocol')
        for _, row in episode.iloc[:count].iterrows():
            state, action, disturbance, target = [np.asarray(json.loads(row[k]), dtype=float)
                for k in ('state', 'action_applied', 'exogenous_available_at_decision', 'next_state')]
            if any(not np.isfinite(value).all() for value in (state, action, disturbance, target)):
                raise ValueError('Nonfinite evaluation data')
            power = float(action[0])
            records.append(dict(episode_id=episode_id, scenario_id=row.scenario_id,
                environment_seed=int(row.seed), source_step=int(row.source_step), policy=row.policy,
                initial_air_c=initial_air, initial_air_bin=initial_bin, power_kw=power,
                action_bin='<1' if power < 1 else ('1-4' if power < 4 else '>=4'),
                state=state, action=action, disturbance=disturbance, target=target))
    if len(splits['test']) != config['origins']['expected_test_episodes']:
        raise ValueError('Test episode count differs from protocol')
    rows = []; manifest = []
    for spec in config['models']:
        model = load_dynamics(spec, root, device, hashes=config['sha256'])
        family = 'residual_mlp' if spec['name'].startswith('residual_') else spec['kind']
        checkpoint_hash = config['sha256'].get(spec.get('checkpoint'))
        manifest.append(dict(**spec, checkpoint_sha256=checkpoint_hash, requested_device=device))
        for record in records:
            # 每个测试起点独立重置，不能在窗口间携带记忆。
            error = ''; predicted = np.full(2, np.nan)
            try:
                model.reset()
                predicted = np.asarray(model.predict_next(record['state'], record['action'], record['disturbance']))
                if predicted.shape != (2,) or not np.isfinite(predicted).all():
                    raise ValueError('Invalid model output')
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                predicted = np.full(2, np.nan)
            metadata = {k: v for k, v in record.items() if k not in ('state', 'action', 'disturbance', 'target')}
            for index, variable in enumerate(('air', 'wall')):
                rows.append(dict(**metadata, model=spec['name'], family=family, seed=spec.get('seed', -1),
                    variable=variable, actual=float(record['target'][index]), predicted=float(predicted[index]),
                    failure_reason=error, horizon=1, mode='one_step',
                    information='current observed outdoor and known load', unit='Celsius',
                    checkpoint_sha256=checkpoint_hash,
                    state=json.dumps(record['state'].tolist()), action=json.dumps(record['action'].tolist()),
                    exogenous=json.dumps(record['disturbance'].tolist())))
    predictions = pd.DataFrame(rows)
    episodes, summary, seeds = summarize_one_step(predictions)
    output = root/config['output']/'one_step'
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output/'predictions.csv.gz', index=False, compression='gzip')
    episodes.to_csv(output/'metrics_by_episode.csv', index=False)
    summary.to_csv(output/'metrics_by_condition.csv', index=False)
    seeds.to_csv(output/'metrics_across_seeds.csv', index=False)
    (output/'model_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    snapshot = dict(config=config, device=device, origins=len(records),
        grouping=dict(initial_air_c=['<19', '[19,21)', '>=21'], power_kw=['<1', '[1,4)', '>=4']),
        single_step_information='Both disturbance conditions coincide at horizon 1; evaluated once.',
        dispersion='sample SD; NaN for fewer than two units',
        failed_predictions='retain denominator; errors on finite predictions only')
    (output/'protocol_snapshot.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    print(summary[summary.grouping == 'all'].to_string(index=False))
    print(f'Saved {len(records)} origins x {len(manifest)} models: {output}')
    return predictions, episodes, summary, seeds


if __name__ == "__main__": main()
