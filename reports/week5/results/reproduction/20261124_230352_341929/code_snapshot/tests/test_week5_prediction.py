"""单步评测统计测试：验证 episode 等权、失败分母及训练种子分离。"""
import unittest
import numpy as np
import pandas as pd
from evaluators.prediction import summarize_one_step
from metrics.prediction import error_by_variable


class OneStepEvaluationTest(unittest.TestCase):
    """用可手算的不同回合长度检查汇总，防止长回合主导均值。"""
    def frame(self):
        rows = []
        for episode, errors in [('a', [1., 1., 1.]), ('b', [3.])]:
            for error in errors:
                rows.append(dict(model='mlp42', family='mlp', seed=42, variable='air',
                    episode_id=episode, policy='random', initial_air_bin='<19', action_bin='1-4',
                    actual=0., predicted=error))
        return pd.DataFrame(rows)

    def test_episode_weighting(self):
        _, summary, seeds = summarize_one_step(self.frame())
        overall = summary[summary.grouping == 'all'].iloc[0]
        self.assertAlmostEqual(overall.mae_c_mean, 2.)
        self.assertAlmostEqual(overall.rmse_c_mean, 2.)
        self.assertAlmostEqual(overall.rmse_c_episode_sd, np.sqrt(2.))
        self.assertEqual(overall.requested, 4)
        self.assertTrue(seeds.rmse_c_seed_sd.isna().all())

    def test_failed_episode_retained(self):
        frame = self.frame()
        frame.loc[frame.episode_id == 'b', 'predicted'] = np.nan
        episodes, summary, _ = summarize_one_step(frame)
        overall = summary[summary.grouping == 'all'].iloc[0]
        self.assertEqual(overall.episodes, 2)
        self.assertEqual(overall.episodes_with_valid_error, 1)
        self.assertEqual(overall.failed, 1)
        self.assertAlmostEqual(overall.valid_fraction, .75)
        self.assertTrue(episodes[(episodes.episode_id == 'b')].rmse_c.isna().all())

    def test_training_seed_sd_separate(self):
        frame = self.frame()
        other = frame.copy()
        other['model'] = 'mlp43'; other['seed'] = 43; other['predicted'] += 2
        _, _, seeds = summarize_one_step(pd.concat([frame, other]))
        overall = seeds[seeds.grouping == 'all'].iloc[0]
        self.assertEqual(overall.runs, 2)
        self.assertAlmostEqual(overall.mae_c_mean, 3.)
        self.assertAlmostEqual(overall.mae_c_seed_sd, np.sqrt(2.))

    def test_variables_have_independent_metrics(self):
        a, r = error_by_variable(np.zeros((2, 2)), np.array([[1., 3.], [1., 3.]]))
        np.testing.assert_equal(a, [1., 3.])
        np.testing.assert_equal(r, [1., 3.])


if __name__ == '__main__':
    unittest.main()
