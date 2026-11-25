"""验证自由递推的信息隔离、天气预测条件和失败前缀指标。"""
import unittest
import numpy as np
from evaluators.rollout import rollout_episode, rollout_errors


class ToyModel:
    """已知增量系统，便于手算长期误差与信息集。"""
    def reset(self):
        pass

    def predict_next(self, state, action, exogenous):
        return state+action[..., :1]+exogenous[..., :1]


class RolloutTest(unittest.TestCase):
    def arrays(self):
        x = np.repeat(np.arange(5.)[:, None], 2, axis=1)
        return x, np.ones((5, 1)), np.zeros((5, 2)), x+1

    def test_free_does_not_read_future_state(self):
        x, u, d, y = self.arrays()
        a = rollout_episode(ToyModel(), x, u, d, y, 5, 'free_running', 'recorded_observation')
        x[1:] = 90
        b = rollout_episode(ToyModel(), x, u, d, y, 5, 'free_running', 'recorded_observation')
        teacher = rollout_episode(ToyModel(), x, u, d, y, 5, 'teacher_forcing', 'recorded_observation')
        np.testing.assert_equal(a['predicted'], b['predicted'])
        self.assertEqual(teacher['predicted'][0, 1, 0], 91)
        np.testing.assert_equal(teacher['predicted'][:, 0], b['predicted'][:, 0])

    def test_future_outdoor_hidden_in_persistence_condition(self):
        x, u, d, y = self.arrays()
        d[1:, 0] = 5
        held = rollout_episode(ToyModel(), x, u, d, y, 5, 'free_running', 'persistence_forecast')
        known = rollout_episode(ToyModel(), x, u, d, y, 5, 'free_running', 'recorded_observation')
        self.assertEqual(held['predicted'][0, -1, 0], 5)
        self.assertEqual(known['predicted'][0, -1, 0], 25)

    def test_divergence_stops_and_retains_prefix(self):
        x, u, d, y = self.arrays()
        result = rollout_episode(ToyModel(), x, u, d, y, 5, 'free_running', 'recorded_observation', (-2, 2))
        self.assertEqual(result['failure_step'][0], 3)
        self.assertEqual(result['predicted'][0, 2, 0], 3)
        self.assertTrue(np.isnan(result['predicted'][0, 3:]).all())

    def test_prefix_definitions_and_denominators(self):
        actual = np.zeros((2, 3, 2))
        predicted = np.ones_like(actual)
        predicted[1, 1:] = np.nan
        rows = {r['definition']: r for r in rollout_errors(actual, predicted, 3) if r['variable'] == 'air'}
        self.assertEqual(rows['terminal']['valid_origins'], 1)
        self.assertEqual(rows['prefix_complete']['valid_points'], 3)
        self.assertEqual(rows['prefix_available']['valid_points'], 4)
        self.assertEqual(rows['prefix_available']['requested'], 2)
        self.assertEqual(rows['prefix_available']['complete_origins'], 1)
        self.assertEqual(rows['prefix_complete']['rmse_c'], 1.)


if __name__ == '__main__':
    unittest.main()
