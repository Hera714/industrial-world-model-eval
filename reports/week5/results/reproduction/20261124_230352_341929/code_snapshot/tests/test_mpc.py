# 总说明：验证动作范围、随机复现、规则回退、超时和完整闭环。
import unittest
from unittest.mock import patch
import numpy as np
from controllers.mpc import MPCConfig, RandomShootingMPC
from simulators.simple_thermal import SimpleThermalEnv
class MPCTest(unittest.TestCase):
    def test_reproducibility(self):
        a=RandomShootingMPC(); b=RandomShootingMPC()
        for _ in range(4): np.testing.assert_array_equal(a.act([19,19,5])[0],b.act([19,19,5])[0])
    def test_no_candidates_fallback(self):
        c=RandomShootingMPC()
        with patch.object(c,'_plan',return_value=(None,'no_valid_candidates')):
            a,info=c.act([19,19,5])
        self.assertEqual(float(a[0]),5.0); self.assertEqual(info['fallback_reason'],'no_valid_candidates')
    def test_actual_nonfinite_candidates(self):
        c=RandomShootingMPC()
        with patch.object(c.rng if False else c,'_plan',wraps=c._plan):
            a,info=c.act([1e308,1e308,1e308])
        self.assertEqual(info['fallback_reason'],'no_valid_candidates'); self.assertEqual(float(a[0]),0)
    def test_timeout(self):
        times=iter([0.0,2.0,2.0]); c=RandomShootingMPC(config=MPCConfig(time_budget_s=1),clock=lambda:next(times))
        a,info=c.act([19,19,5]); self.assertEqual(info['fallback_reason'],'timeout'); self.assertEqual(float(a[0]),5)
    def test_invalid_observation(self):
        a,info=RandomShootingMPC().act([float('nan'),20,5]); self.assertEqual(info['fallback_reason'],'invalid_observation'); self.assertEqual(float(a[0]),0)
    def test_episode(self):
        env=SimpleThermalEnv(); c=RandomShootingMPC(); obs,_=env.reset(seed=42)
        for i in range(144):
            a,info=c.act(obs); self.assertTrue(env.action_space.contains(a)); self.assertGreaterEqual(info['planning_time_s'],0)
            obs,_,term,trunc,_=env.step(a)
        self.assertFalse(term); self.assertTrue(trunc)
if __name__=='__main__': unittest.main()
