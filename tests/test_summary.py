"""统计模块验收：两层 SD、同回合配对、失败保留与缺失/重复输入检查。"""
import unittest
import numpy as np
import pandas as pd
from evaluators.summary import summarize_episode_metrics,episode_seed_metrics,paired_episode_differences,summarize_pairs


class SummaryTest(unittest.TestCase):
    def frame(self):
        rows=[]
        for seed in (42,43,44):
            for ep in (0,1):
                value=1+2*(seed-42)+2*ep
                rows.append(dict(model=f'mlp{seed}',family='mlp',seed=seed,episode_id=ep,
                    mode='free_running',condition='recorded',horizon=5,definition='terminal',variable='air',
                    unit='Celsius',requested=10,valid_origins=10,complete_origins=10,valid_points=10,
                    failed_origins=0,mae_c=value,rmse_c=value))
        for ep in (0,1):
            row=rows[0].copy()
            row.update(model='baseline',family='baseline',seed=-1,episode_id=ep,mae_c=2.,rmse_c=2.)
            rows.append(row)
        return pd.DataFrame(rows)

    def test_two_level_statistics(self):
        summary,seeds=summarize_episode_metrics(self.frame())
        run=summary[summary.model.eq('mlp42')].iloc[0]
        self.assertAlmostEqual(run.rmse_c_mean,2.)
        self.assertAlmostEqual(run.rmse_c_episode_sd,np.sqrt(2))
        row=seeds[seeds.family.eq('mlp')].iloc[0]
        self.assertEqual(row.runs,3)
        self.assertEqual(row.rmse_c_mean,4.)
        self.assertEqual(row.rmse_c_seed_sd,2.)
        self.assertTrue(np.isnan(seeds[seeds.family.eq('baseline')].iloc[0].rmse_c_seed_sd))

    def test_episode_pair_direction(self):
        episodes=episode_seed_metrics(self.frame())
        pairs=paired_episode_differences(episodes,{'baseline':'current','mlp':'current'})
        np.testing.assert_allclose(pairs.sort_values('episode_id').delta_rmse_c,[-1.,-3.])
        aggregate=summarize_pairs(pairs).iloc[0]
        self.assertEqual(aggregate.requested_episode_pairs,2)
        self.assertEqual(aggregate.delta_rmse_c_mean,-2.)
        self.assertAlmostEqual(aggregate.delta_rmse_c_episode_sd,np.sqrt(2))

    def test_failed_support_retains_pair(self):
        frame=self.frame()
        mask=frame.model.eq('mlp42')&frame.episode_id.eq(0)
        frame.loc[mask,['failed_origins','valid_origins','complete_origins']]=[1,9,9]
        pairs=paired_episode_differences(episode_seed_metrics(frame))
        self.assertEqual(len(pairs),2)
        self.assertTrue(np.isnan(pairs[pairs.episode_id.eq(0)].iloc[0].delta_rmse_c))
        summary=summarize_pairs(pairs).iloc[0]
        self.assertEqual(summary.invalid_episode_pairs,1)

    def test_missing_and_duplicate_episode_rows_rejected(self):
        frame=self.frame()
        with self.assertRaises(ValueError): summarize_episode_metrics(frame.iloc[1:])
        with self.assertRaises(ValueError): summarize_episode_metrics(pd.concat([frame,frame.iloc[:1]]))

    def test_outer_pair_retains_missing_episode(self):
        episodes=episode_seed_metrics(self.frame())
        episodes=episodes[~(episodes.family.eq('baseline')&episodes.episode_id.eq(1))]
        pairs=paired_episode_differences(episodes)
        self.assertEqual(len(pairs),2)
        self.assertEqual(pairs[pairs.episode_id.eq(1)].iloc[0].pair_status,'missing_episode')

    def test_different_origin_counts_do_not_form_valid_pair(self):
        episodes=episode_seed_metrics(self.frame())
        episodes.loc[episodes.family.eq('baseline'),'requested']=11
        pairs=paired_episode_differences(episodes)
        self.assertFalse(pairs.paired_valid.any())
        self.assertTrue(pairs.delta_rmse_c.isna().all())

    def test_crps_reuses_episode_statistics_and_pairing(self):
        frame=self.frame().drop(columns='rmse_c').rename(columns={'mae_c':'empirical_crps_c'})
        summary,seeds=summarize_episode_metrics(frame)
        self.assertEqual(seeds[seeds.family.eq('mlp')].iloc[0].empirical_crps_c_mean,4.)
        paired=paired_episode_differences(episode_seed_metrics(frame))
        result=summarize_pairs(paired).iloc[0]
        self.assertEqual(result.delta_empirical_crps_c_mean,-2.)


if __name__=='__main__': unittest.main()
