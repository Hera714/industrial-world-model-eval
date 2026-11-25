"""绘制 Week 3 seed42 随机策略轨迹，展示观测、增量动作和成本。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
folder = root / 'reports/week3/results/industrial_benchmark/baseline'
rows = [json.loads(line) for line in (folder / 'episode_42.jsonl').read_text().splitlines()]
steps = np.array([row['step'] for row in rows])
fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
# 公开操纵量采用相同基准尺度。
for name in ['velocity', 'gain', 'shift']:
    axes[0].plot(steps, [row['state'][name] for row in rows], label=name, linewidth=1.1)
axes[0].axhline(50, color='gray', linestyle='--', label='setpoint = 50')
axes[0].set(ylabel='Benchmark scale', title='Public observations: steering variables')
axes[0].legend(ncol=4)
for name in ['fatigue', 'consumption']:
    axes[1].plot(steps, [row['state'][name] for row in rows], label=name, linewidth=1)
axes[1].set(ylabel='Benchmark index', title='Fatigue and consumption')
axes[1].legend(ncol=2)
# 三个动作在各自纵向偏移的轨道显示，便于辨认随机增量。
actions = np.array([row['action'] for row in rows])
for index, name in enumerate(['delta_velocity', 'delta_gain', 'delta_shift']):
    axes[2].plot(steps, actions[:, index] + 3 * index, linewidth=0.45, label=name)
axes[2].set_yticks([0,3,6], ['velocity (±1)', 'gain (±1)', 'shift (±1)'])
axes[2].set(title='Uniform random actions: vertically offset for readability')
cost = np.array([row['cost'] for row in rows])
axes[3].plot(steps, cost, color='tab:red', alpha=.65, linewidth=.8, label='step cost')
axes[3].axhline(cost.mean(), color='black', linestyle='--', label=f'mean = {cost.mean():.2f}')
axes[3].set(ylabel='Benchmark cost', xlabel='Simulation step', title='Native cost = 3 × fatigue + consumption')
axes[3].legend()
for axis in axes:
    axis.grid(alpha=.2)
    axis.set_xlim(1,1000)
fig.suptitle('Industrial Benchmark | seed 42 | uniform random policy | setpoint 50', fontsize=15)
fig.savefig(folder / 'seed42_trajectory.png', dpi=160)
plt.close(fig)
