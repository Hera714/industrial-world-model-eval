"""把 Siemens IB 的旧 Gym 接口转换成可复现的 Gymnasium 环境。"""
from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
import numpy as np
from gymnasium import spaces
from simulators.base_env import BaseEnv

_RNG_LOCK = RLock()


class IndustrialBenchmarkEnv(BaseEnv):
    """三维增量动作，六维公开观测，时间单位为 step。"""
    metadata = {'render_modes': []}

    def __init__(self, setpoint=50, max_episode_steps=1000):
        super().__init__()
        if not np.isfinite(setpoint) or not 0 <= setpoint <= 100:
            raise ValueError('setpoint must be finite and in [0, 100]')
        if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int) or max_episode_steps < 1:
            raise ValueError('max_episode_steps must be a positive integer')
        self.setpoint = float(setpoint)
        self.max_episode_steps = max_episode_steps
        self.action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(np.zeros(6, dtype=np.float32), np.array([100, 100, 100, 100, 1000, 1000], dtype=np.float32))
        self._done = True
        self._upstream = None

    @contextmanager
    def _random_context(self):
        """上游使用全局 NumPy 随机源，逐实例保存并恢复以支持交错运行。"""
        with _RNG_LOCK:
            outside = np.random.get_state()
            np.random.set_state(self._rng_state)
            try:
                yield
            finally:
                self._rng_state = np.random.get_state()
                np.random.set_state(outside)

    def _info(self):
        """内部隐藏状态只放诊断字段，公开观测保持六维。"""
        return {'step': self.step_count, 'time_unit': 'step', 'cost': float(self._upstream.IB.state['cost']), 'upstream_info': deepcopy(self._upstream.info)}

    def reset(self, *, seed=None, options=None):
        """显式种子重建上游；无种子重置从 Gymnasium 随机源派生新种子。"""
        options = options or {}
        if set(options) - {'setpoint'}:
            raise ValueError('Unknown reset option')
        setpoint = options.get('setpoint', self.setpoint)
        if not np.isfinite(setpoint) or not 0 <= setpoint <= 100:
            raise ValueError('setpoint must be finite and in [0, 100]')
        super().reset(seed=seed)
        upstream_seed = int(self.np_random.integers(0, 2**31))
        self._rng_state = np.random.RandomState(upstream_seed).get_state()
        # 上游旧版别名的等价兼容，导入后执行原始动力学。
        with self._random_context():
            np.float = float
            np.int = int
            from industrial_benchmark_python.IBGym import IBGym
            self._upstream = IBGym(setpoint, init_seed=upstream_seed, reset_after_timesteps=self.max_episode_steps)
        self.step_count = 0
        self._done = False
        return np.asarray(self._upstream.observation, dtype=np.float32).copy(), self._info()

    def step(self, action):
        """先校验动作，推进一次上游；时间上限映射为 truncated。"""
        if self._done:
            raise RuntimeError('Call reset before step or after episode end')
        try:
            action = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError('Action must be three finite values in [-1, 1]') from error
        if action.shape != (3,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Action must be three finite values in [-1, 1]')
        try:
            with self._random_context():
                observation, reward, done, _ = self._upstream.step(action)
        except Exception:
            self._done = True
            raise
        self.step_count += 1
        self._done = bool(done)
        info = self._info()
        if done:
            info['end_reason'] = 'time_limit'
        return np.asarray(observation, dtype=np.float32).copy(), float(reward), False, bool(done), info
