"""验证 IB 适配器的复现、非法动作拒绝和时间上限契约。"""
import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from simulators.industrial_benchmark import IndustrialBenchmarkEnv


class TestIndustrialBenchmark(unittest.TestCase):
    def test_contract(self):
        # 检查标准接口和固定种子确定性。
        check_env(IndustrialBenchmarkEnv(), skip_render_check=True)

    def test_interleaved_and_end(self):
        # 两个实例交错推进也应得到相同轨迹。
        first = IndustrialBenchmarkEnv(max_episode_steps=3)
        second = IndustrialBenchmarkEnv(max_episode_steps=3)
        np.testing.assert_array_equal(first.reset(seed=42)[0], second.reset(seed=42)[0])
        for index in range(3):
            left = first.step([0.1, -0.2, 0.3])
            right = second.step([0.1, -0.2, 0.3])
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1:4], right[1:4])
            self.assertFalse(left[2])
            self.assertEqual(left[3], index == 2)
            self.assertEqual(left[4]['step'], index + 1)
            self.assertEqual(left[4].get('end_reason'), 'time_limit' if index == 2 else None)
        with self.assertRaisesRegex(RuntimeError, 'reset'):
            first.step([0, 0, 0])

    def test_boundary_action_after_reset(self):
        # 回合结束后重置，边界动作仍应正常推进并应用实际增量。
        env = IndustrialBenchmarkEnv(max_episode_steps=2)
        env.reset(seed=42)
        env.step([0, 0, 0])
        self.assertTrue(env.step([0, 0, 0])[3])
        initial, info = env.reset(seed=42)
        self.assertEqual(info['step'], 0)
        action = np.array([-1, 1, 0], dtype=np.float32)
        self.assertTrue(env.action_space.contains(action))
        observation, reward, terminated, truncated, info = env.step(action)
        np.testing.assert_allclose(observation[1:4], initial[1:4] + [-1, 10, 0])
        self.assertEqual(reward, -info['cost'])
        self.assertEqual(info['step'], 1)
        self.assertFalse(terminated or truncated)

    def test_invalid_actions(self):
        # 非法动作不得推进环境。
        env = IndustrialBenchmarkEnv()
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0])
        env.reset(seed=42)
        for action in ([2, 0, 0], [0, -2, 0], [np.nan, 0, 0], [0, 0, np.inf], [0, 0], ['bad', 0, 0]):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.step_count, 0)
