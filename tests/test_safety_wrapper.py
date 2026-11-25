# 总说明：验证安全包装层确实执行后备动作、保存事件并在真实状态越界时终止。
import unittest
import numpy as np
from simulators.simple_thermal import SimpleThermalEnv, ThermalConfig
from simulators.safety import ThermalSafetyWrapper

class SafetyWrapperTest(unittest.TestCase):
    def test_invalid_action_logged(self):
        e=ThermalSafetyWrapper(SimpleThermalEnv()); e.reset(seed=1)
        obs,reward,term,trunc,info=e.step([float('nan')])
        self.assertEqual(info['action_kw'],0); self.assertEqual(info['safety_reason'],'invalid_action')
        self.assertEqual(len(e.events),1)
    def test_high_observed_action_override(self):
        e=ThermalSafetyWrapper(SimpleThermalEnv()); e.reset(seed=1,options={'initial_air_c':31})
        obs,reward,term,trunc,info=e.step([5])
        self.assertEqual(info['action_kw'],0); self.assertEqual(info['safety_reason'],'high_temperature')
        self.assertTrue(term); self.assertEqual(info['environment_safety_reason'],'true_high_boundary')
        with self.assertRaises(RuntimeError): e.step([0])
    def test_low_state_terminates(self):
        c=ThermalConfig(measurement_noise_std=0,process_noise_std=0)
        e=ThermalSafetyWrapper(SimpleThermalEnv(c)); e.reset(seed=1,options={'initial_air_c':5})
        _,_,term,_,info=e.step([0])
        self.assertEqual(info['safety_reason'],'low_temperature'); self.assertEqual(info['action_kw'],5); self.assertTrue(term)
if __name__=='__main__': unittest.main()
