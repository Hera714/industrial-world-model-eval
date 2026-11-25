"""采集动作覆盖轨迹，保存分组切分清单、原始配置与分布图。"""
from pathlib import Path
import sys
import json
import hashlib
from dataclasses import asdict
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.collect_trajectories import collect_episodes
from data.split import split_by_episode
from simulators.simple_thermal import ThermalConfig


def main():
    """从固定配置运行完整采集，并按策略分层分配回合。"""
    config = yaml.safe_load((ROOT/'configs/experiments/week4_thermal.yaml').read_text())
    env_config = ThermalConfig(episode_steps=config['episode_steps'])
    frame = collect_episodes(config['n_episodes'], config['base_seed'], env_config, tuple(config['policies']))
    for _, episode in frame.groupby('episode_id'):
        assert len(episode) == config['episode_steps']
        assert episode.source_step.tolist() == list(range(config['episode_steps']))
        assert episode.iloc[-1].truncated
    parts, manifest = split_by_episode(frame, config['split_seed'])
    _, repeat = split_by_episode(frame, config['split_seed'])
    assert manifest == repeat
    output, reports = ROOT/config['output'], ROOT/config['reports']
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output/'trajectories.csv', index=False)
    for name, part in parts.items():
        part.to_csv(output/f'{name}.csv', index=False)
    metadata = dict(config=config, environment=asdict(env_config), splits=manifest,
        sha256=hashlib.sha256((output/'trajectories.csv').read_bytes()).hexdigest(),
        counts={name:dict(episodes=len(manifest[name]), rows=len(part), policies=part.groupby('policy').episode_id.nunique().to_dict()) for name,part in parts.items()})
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2))
    # 相同区间比较分布，额外显示初态范围和三个策略的动作覆盖。
    fig, axes = plt.subplots(1, 3, figsize=(15,4), constrained_layout=True)
    bins = np.linspace(0, env_config.max_power_kw, 21)
    for policy, group in frame.groupby('policy'):
        axes[0].hist(group.action, bins=bins, histtype='step', density=True, label=policy)
    for name, part in parts.items():
        axes[1].hist(part.action, bins=bins, histtype='step', density=True, label=name)
    initial = frame[frame.source_step == 0]
    axes[2].scatter(initial.air, initial.wall, s=12)
    axes[0].set(xlabel='Applied heating power (kW)', ylabel='Density', title='Action coverage by policy')
    axes[1].set(xlabel='Applied heating power (kW)', ylabel='Density', title='Action coverage by split')
    axes[2].set(xlabel='Initial air temperature (°C)', ylabel='Initial wall temperature (°C)', title='Initial states')
    for axis in axes[:2]:
        axis.legend()
    fig.savefig(reports/'action_distribution.png', dpi=150)
    plt.close(fig)
    print(json.dumps(metadata['counts'], indent=2))


if __name__ == '__main__':
    main()
