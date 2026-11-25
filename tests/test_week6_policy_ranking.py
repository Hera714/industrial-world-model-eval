"""总说明：用手算成本核对排序反转、无效真实最佳计划、打平与无可选计划。"""
import unittest
import numpy as np
from evaluators.consistency import score_fixed_action_plans


class RankingTest(unittest.TestCase):
    def plans(self):
        return [dict(policy=n,true_cost=t,predicted_cost=p,valid=True) for n,t,p in [('A',0.,5.),('B',2.,0.),('C',5.,3.)]]

    def test_reversal_and_regret(self):
        result=score_fixed_action_plans(self.plans())
        self.assertEqual(result['concordant'],1)
        self.assertEqual(result['discordant'],2)
        self.assertAlmostEqual(result['direction_agreement'],1/3)
        self.assertEqual(result['selected_policy'],'B')
        self.assertEqual(result['regret_vs_all_true_policies'],2.)

    def test_invalid_best_remains_in_true_regret_baseline(self):
        plans=self.plans();plans[0]['valid']=False
        result=score_fixed_action_plans(plans)
        self.assertEqual(result['invalid_pairs'],2)
        self.assertEqual(result['direction_agreement'],1.)
        self.assertEqual(result['regret_vs_all_true_policies'],2.)

    def test_all_invalid_has_no_invented_zero_regret(self):
        plans=self.plans()
        for row in plans:row['valid']=False
        result=score_fixed_action_plans(plans)
        self.assertIsNone(result['selected_policy'])
        self.assertTrue(np.isnan(result['regret_vs_all_true_policies']))

    def test_prediction_tie_is_counted(self):
        plans=self.plans();plans[0]['predicted_cost']=0.
        result=score_fixed_action_plans(plans)
        self.assertEqual(result['tied'],1)


if __name__=='__main__':unittest.main()
