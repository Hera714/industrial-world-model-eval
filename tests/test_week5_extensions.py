"""验证失败计数不等同于缺失计数，及 IB 自由递推的历史信息边界。"""
import unittest
import numpy as np
import pandas as pd
from evaluators.rollout import summarize_rollout_metrics
from scripts.evaluate_week5_ib import ib_window


class ExtensionTest(unittest.TestCase):
    def test_failure_count_and_seed_summary(self):
        rows=[]
        for seed in (42,43,44):
            for episode in (0,1):
                rows.append(dict(model=f'mlp{seed}',family='mlp',seed=seed,mode='free_running',
                    condition='recorded',horizon=2,definition='terminal',variable='air',episode_id=episode,
                    mae_c=1.,rmse_c=1.,requested=10,valid_origins=10,complete_origins=10,
                    valid_points=10,failed_origins=1))
        summary,seeds=summarize_rollout_metrics(pd.DataFrame(rows))
        self.assertTrue((summary.failed_origins==2).all())
        self.assertTrue((summary.unavailable_origins==0).all())
        self.assertEqual(seeds.iloc[0].requested_per_run,20)
        self.assertEqual(seeds.iloc[0].failed_model_origins,6)
        self.assertEqual(seeds.iloc[0].valid_fraction_mean,1.)
        self.assertAlmostEqual(seeds.iloc[0].failure_fraction_mean,.1)
        self.assertEqual(seeds.iloc[0].runs,3)

    def test_ib_free_window_does_not_read_future_observations(self):
        obs=np.ones((40,6)); actions=np.zeros((39,3)); window=np.zeros((1,20,9))
        a=ib_window(obs,actions,np.array([19]),1,'free_running',window)
        obs[20:]=999
        b=ib_window(obs,actions,np.array([19]),1,'free_running',window)
        np.testing.assert_equal(a,b)
        teacher=ib_window(obs,actions,np.array([19]),1,'teacher_forcing',window)
        self.assertEqual(teacher[0,-1,0],999)


if __name__=='__main__': unittest.main()
