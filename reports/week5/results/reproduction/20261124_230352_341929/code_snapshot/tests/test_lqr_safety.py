# 总说明：验证 LQR 稳态、动作限幅以及安全覆盖和非法输入记录。
import unittest
import numpy as np
from controllers.lqr import ThermalLQR
from controllers.safety import SafetyGuard
from simulators.simple_thermal import SimpleThermalEnv
from controllers.mpc import RandomShootingMPC, MPCConfig

class SafetyLQRTest(unittest.TestCase):
    def test_lqr_equilibrium_and_clip(self):
        m=ThermalLQR(); c=m.c; air,wall=m.x_eq; d=m.outdoor_equilibrium_c
        next_air=air+c.dt_hours/c.air_capacity*(c.air_wall_conductance*(wall-air)+c.air_out_conductance*(d-air)+m.u_eq+c.internal_load_kw)
        next_wall=wall+c.dt_hours/c.wall_capacity*(c.air_wall_conductance*(air-wall)+c.wall_out_conductance*(d-wall))
        np.testing.assert_allclose([next_air,next_wall],m.x_eq,atol=1e-12)
        action,detail=m.act([0,0,6]); self.assertTrue(detail['action_clipped']); self.assertEqual(float(action[0]),5.0)
    def test_guard_and_environment(self):
        g=SafetyGuard(); env=SimpleThermalEnv(); env.reset(seed=3)
        for obs,act,expected,reason in [([10,18,6],[0],5,'low_temperature'),([31,18,6],[5],0,'high_temperature'),([21,18,6],[float('nan')],0,'invalid_action'),([21,18,6],[8],5,'action_clipped')]:
            u,info=g.apply(obs,act); self.assertEqual(float(u[0]),expected); self.assertEqual(info['safety_reason'],reason); self.assertTrue(env.action_space.contains(u))
        with self.assertRaises(ValueError): env.step([float('nan')])
    def test_controller_exception_fallback(self):
        class Broken:
            def predict_next(self,*args): raise RuntimeError('deliberate model failure')
        m=RandomShootingMPC(dynamics=Broken()); action,detail=m.act([19,19,6])
        self.assertEqual(float(action[0]),5.0); self.assertEqual(detail['fallback_reason'],'planning_error:RuntimeError')
if __name__=='__main__': unittest.main()
