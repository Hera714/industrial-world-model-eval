"""Week 3 基线实验：固定种子随机策略，导出 IB 轨迹、指标和中文报告。"""
import csv
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluators.closed_loop import run_standard_episode
from simulators.industrial_benchmark import IndustrialBenchmarkEnv


def run_boptest(config):
    """共用回合循环，保留原生测点、版本和期末 KPI。"""
    from simulators.boptest_adapter import BoptestEnv

    output = ROOT / 'reports/week3/results/boptest/baseline'
    env = BoptestEnv(config)
    try:
        def make_row(obs, action, terminated, truncated, info):
            return dict(**info, observation=obs.tolist(), action=int(action),
                        terminated=terminated, truncated=truncated,
                        end_reason='time_limit' if truncated else None)

        rows = run_standard_episode(env, lambda obs: 0, 42, 'embedded', make_row)
        kpi = env.request('GET', f'/kpi/{env.testid}')
        version = env.request('GET', '/version')
        measurements = env.request('GET', f'/measurements/{env.testid}')
    finally:
        env.close()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'episode_42.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (output / 'summary.json').write_text(json.dumps(dict(config=config, version=version,
        measurements=measurements, steps=len(rows), native_kpi=kpi), indent=2))
    print(json.dumps(kpi, indent=2))


def main():
    """采用官方样例的均匀随机动作基线，按完整回合汇总。"""
    output = ROOT / 'reports/week3/results/industrial_benchmark/baseline'
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in [42, 43, 44]:
        env = IndustrialBenchmarkEnv(setpoint=50, max_episode_steps=1000)
        rng = np.random.default_rng(seed)
        rows = run_standard_episode(env, lambda obs: rng.uniform(-1, 1, 3), seed, f'random_{seed}')
        assert len(rows) == 1000 and rows[-1]['truncated']
        # 原生 reward 与成本公式分别核验，确保指标口径一致。
        costs = np.array([row['cost'] for row in rows])
        rewards = np.array([row['reward'] for row in rows])
        fatigue = np.array([row['state']['fatigue'] for row in rows])
        consumption = np.array([row['state']['consumption'] for row in rows])
        np.testing.assert_allclose(costs, -rewards)
        np.testing.assert_allclose(costs, 3 * fatigue + consumption, rtol=1e-6)
        with (output / f'episode_{seed}.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, default=lambda value: value.item()) + '\n')
        summaries.append(dict(seed=seed, steps=len(rows), native_return=float(rewards.sum()),
            cumulative_cost=float(costs.sum()), mean_cost=float(costs.mean()),
            mean_fatigue=float(fatigue.mean()), mean_consumption=float(consumption.mean()),
            observation_bound_violations=sum(not row['constraints']['observation_in_declared_space'] for row in rows),
            decision_mean_ms=float(np.mean([row['decision_time_s'] for row in rows]) * 1000),
            end_reason=rows[-1]['end_reason']))
        env.close()
    with (output / 'episode_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0])
        writer.writeheader()
        writer.writerows(summaries)
    # 保存真实软件版本、源码提交和策略信息。
    config = dict(seeds=[42,43,44], setpoint=50, max_episode_steps=1000,
        policy='uniform_random', action_seed='same as episode seed', time_unit='step',
        versions={name: importlib.metadata.version(name) for name in ['industrial-benchmark-python','gym','gymnasium','numpy','scipy']},
        upstream_commit=subprocess.check_output(['git','-C',str(ROOT/'external/industrialbenchmark'),'rev-parse','HEAD'], text=True).strip())
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    aggregates = {key: dict(mean=float(np.mean([row[key] for row in summaries])), std=float(np.std([row[key] for row in summaries], ddof=1))) for key in ['native_return','cumulative_cost','mean_cost','mean_fatigue','mean_consumption']}
    (output / 'summary.json').write_text(json.dumps(aggregates, indent=2))
    report = ['# Week 3 工业模拟器基线实验', '',
        '使用官方样例同类的均匀随机策略，每步独立采样三个 [-1,1] 增量动作。工况为 50，每回合 1000 步，环境与策略种子均为 42、43、44。', '',
        '| 种子 | 步数 | 原生累计奖励 | 累计成本 | 平均成本 | 平均疲劳 | 平均消耗 | 结束原因 |',
        '|---|---|---|---|---|---|---|---|']
    for row in summaries:
        report.append(f"| {row['seed']} | {row['steps']} | {row['native_return']:.3f} | {row['cumulative_cost']:.3f} | {row['mean_cost']:.3f} | {row['mean_fatigue']:.3f} | {row['mean_consumption']:.3f} | {row['end_reason']} |")
    report += ['', '## 三回合汇总', '', '均值 ± 样本标准差（ddof=1），实验单位为完整回合。', '']
    report += [f"- {key} 为 {value['mean']:.3f} ± {value['std']:.3f}" for key,value in aggregates.items()]
    report += ['', '## 指标口径与复现', '',
        '原生逐步成本为 3 × fatigue + consumption，classic 奖励为其相反数。累计成本、平均成本、平均疲劳和平均消耗由逐步输出计算，保留上游基准尺度。原生累计奖励等于累计成本的相反数。', '',
        '官方示例统计每个 100 步片段的折扣奖励（折扣 0.97），并除以 100。本报告统计完整 1000 步无折扣累计奖励，两者统计区间和折扣定义不同。', '',
        '运行命令为 `python scripts/run_week3.py`。配置、软件版本和上游提交见 results/industrial_benchmark/baseline/config.json；逐回合 CSV、汇总 JSON 和三份 JSONL 轨迹位于同目录。', '',
        '所有回合按时间上限截断。日志保留动作前观测、动作后公开状态、原生奖励、成本、诊断字段和空间检查结果。']
    (ROOT / 'reports/week3.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if args.config:
        import yaml

        config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        if config['environment'] == 'boptest':
            run_boptest(config)
        elif config['environment'] == 'industrial_benchmark':
            main()
        else:
            parser.error('Unknown environment')
    else:
        main()
