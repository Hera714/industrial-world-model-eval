"""运行 Siemens 官方样例及一个完整回合，保存接口返回、配置和结束原因。"""
import contextlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root / 'reports/week3/results/industrial_benchmark/upstream'
out.mkdir(parents=True, exist_ok=True)
# 兼容上游旧 NumPy 别名，映射到原先等价的内建类型。
np.float = float
np.int = int
from industrial_benchmark_python.IBGym import IBGym

def encode(value):
    """把原始返回值中的 NumPy 类型转换为可追溯 JSON。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

# 原样执行官方测试脚本，单独保存标准输出和警告。
with (out / 'official_compatible_stdout.log').open('w') as stdout, (out / 'official_compatible_stderr.log').open('w') as stderr:
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        runpy.run_path(str(root / 'external/industrialbenchmark/industrial_benchmark_python/test_gym_wrapper.py'), run_name='__main__')

# 固定环境及动作种子，记录 reset 和每个四元组返回。
config = dict(setpoint=50, reward_type='classic', action_type='continuous', observation_type='classic', reset_after_timesteps=1000, init_seed=42)
env = IBGym(**config)
env.action_space.seed(42)
with (out / 'episode.jsonl').open('w') as stream:
    stream.write(json.dumps(dict(step=0, observation=env.observation), default=encode) + '\n')
    for step in range(1, 1001):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        stream.write(json.dumps(dict(step=step, action=action, observation=observation, reward=reward, done=done, info=info), default=encode) + '\n')
        if done:
            break
assert done and step == 1000
summary = dict(config=config, steps=step, done=bool(done), end_reason='reset_after_timesteps', reset_api='observation', step_api=['observation', 'reward', 'done', 'info'], python=sys.version, compatibility_aliases=['np.float=float', 'np.int=int', 'np.bool=bool'], versions={name: importlib.metadata.version(name) for name in ['industrial-benchmark-python', 'gym', 'numpy', 'scipy']}, commit=subprocess.check_output(['git', '-C', str(root / 'external/industrialbenchmark'), 'rev-parse', 'HEAD'], text=True).strip())
summary['compatibility_aliases'] = ['np.float=float', 'np.int=int']
(out / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
