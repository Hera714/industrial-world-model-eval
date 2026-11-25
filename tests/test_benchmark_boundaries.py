"""总说明：验证非法预测保留、历史窗口隔离与观测故障随机流复现。"""
from pathlib import Path
import unittest
import numpy as np
from benchmark.config import load_config
from evaluators.benchmark_boptest import predict_window, corrupt_observation, endpoint_metrics

ROOT = Path(__file__).resolve().parents[1]


class SequenceModel:
    """在指定步返回异常值，观察评测是否保留前缀与失败分母。"""
    length = 16
    def __init__(self, value):
        self.calls = 0
        self.value = value
    def predict(self, history):
        self.calls += 1
        if self.calls == 2:
            return self.value
        return np.array([21., 30., 28., 1., 1., .2])


class BenchmarkBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT/'configs/benchmark/boptest_week6.yaml', ROOT)
        cls.features = np.ones((32, 12))
        cls.features[:, :5] = [21., 30., 28., 1., 1.]

    def test_invalid_output_retains_prefix_and_denominator(self):
        for bad in (np.full(6, np.nan), np.full(6, np.inf), np.full(6, 1001.), np.ones(5)):
            with self.subTest(bad=bad):
                values, valid, _, failure = predict_window(SequenceModel(bad), self.features, 16, 4, self.config, 'clean', 'free_running')
                self.assertEqual(failure['step'], 2)
                np.testing.assert_array_equal(valid, [True, False, False, False])
                self.assertEqual(values[0, 0], 21.)
                if bad.shape == (6,):
                    np.testing.assert_array_equal(values[1], bad)
                result = endpoint_metrics(np.ones((4, 6)), values, valid)
                self.assertEqual((result['requested'], result['valid'], result['failed']), (4, 1, 3))

    def test_seeded_noise_and_missing_reproduce(self):
        original = np.array([21., 30., 28., 1., 1.])
        faults = self.config['evaluator']['faults']
        def sample(seed, condition):
            rng = np.random.default_rng(seed)
            previous = 29.
            result = []
            for step in range(8):
                raw, filled, mask, previous = corrupt_observation(original, condition, faults, rng, previous, step)
                result.append((raw, filled, mask))
            return result
        for condition in ('noise_0p3', 'missing_4'):
            first, again = sample(64042, condition), sample(64042, condition)
            for a, b in zip(first, again):
                for x, y in zip(a, b): np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(sample(64042, 'noise_0p3')[0][0], sample(64043, 'noise_0p3')[0][0]))
        missing = sample(64042, 'missing_4')
        self.assertTrue(np.isnan(missing[0][0][1]))
        self.assertEqual(missing[0][1][1], 29.)
        self.assertFalse(missing[4][2][1])

    def test_window_is_not_mutated(self):
        before = self.features.copy()
        predict_window(SequenceModel(np.ones(6)), self.features, 16, 4, self.config, 'noise_0p3', 'teacher_forcing')
        np.testing.assert_array_equal(self.features, before)


if __name__ == '__main__': unittest.main()
