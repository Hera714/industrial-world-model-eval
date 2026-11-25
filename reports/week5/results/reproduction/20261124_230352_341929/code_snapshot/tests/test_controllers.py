# 总说明：验证滞回控制器的开关阈值、记忆、合法动作和完整闭环。
import unittest
import numpy as np
from controllers.rule_based import HysteresisController
from simulators.simple_thermal import SimpleThermalEnv
class HysteresisTest(unittest.TestCase):
    def test_thresholds_memory_and_reset(self):
        c=HysteresisController()
        def p(t): return float(c.act([t,20,5])[0])
        self.assertEqual(p(20.5),0); self.assertEqual(p(20.4),5); self.assertEqual(p(21.5),5)
        self.assertEqual(p(21.6),0); self.assertEqual(p(21),0); c.reset(True); self.assertEqual(p(21),5)
    def test_invalid_observation(self):
        c=HysteresisController()
        for x in ([float('nan'),1,2],[1,2], [1,2,float('inf')]):
            with self.assertRaises(ValueError): c.act(x)
    def test_full_episode(self):
        e=SimpleThermalEnv(); c=HysteresisController(heating_power_kw=e.config.max_power_kw); obs,_=e.reset(seed=42)
        for i in range(144):
            action=c.act(obs); self.assertTrue(e.action_space.contains(action))
            obs,reward,terminated,truncated,info=e.step(action)
            self.assertFalse(terminated); self.assertEqual(truncated,i==143); self.assertTrue(np.isfinite(info['cost']))
if __name__=='__main__': unittest.main()
