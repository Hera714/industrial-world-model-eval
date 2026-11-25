"""总说明：模拟服务端核对覆盖、释放、非法值与 fallback；真实 API 验收另行保存。"""
import unittest
import numpy as np
from simulators.boptest_control import BoptestControlEnv


class FakeControl(BoptestControlEnv):
    def __init__(self):
        super().__init__(dict(url='http://127.0.0.1',measurements=['reaTZon_y'],start_time=0,step_seconds=900,episode_steps=2,
            action=dict(point='oveTSet_u',activation='oveTSet_activate',experiment_bounds_c=[18,24])))
        self.done=False; self.testid='fake'; self.steps=0; self.commands=[]; self.active=False
    def request(self,method,path,data=None):
        if path.startswith('/advance/'):
            self.commands.append(data); self.active=bool(data['oveTSet_activate'])
            self.applied=data.get('oveTSet_u',294.15)
            return dict(time=(self.steps+1)*900,reaTZon_y=294.15)
        if path.startswith('/kpi/'): return dict(ener_tot=(self.steps+1)*.001)
        if path.startswith('/results/'): return dict(oveTSet_u=[self.applied])
        return None


class ControlContractTest(unittest.TestCase):
    def test_override_release_end(self):
        env=FakeControl(); result=env.step([22]); self.assertTrue(env.active)
        self.assertAlmostEqual(result[-1]['applied_action_c'],22)
        result=env.step(None); self.assertFalse(env.active); self.assertTrue(result[3]); self.assertFalse(result[2])
        with self.assertRaises(RuntimeError): env.step([21])
    def test_invalid_does_not_advance(self):
        env=FakeControl()
        for action in ([17],[25],[float('nan')],[float('inf')],[],[20,21]):
            with self.assertRaises(ValueError): env.step(action)
        self.assertEqual(env.steps,0); self.assertEqual(env.commands,[])
    def test_fallback_releases(self):
        env=FakeControl(); result=env.safe_step([float('nan')])
        self.assertFalse(env.active); self.assertEqual(result[-1]['fallback_reason'],'invalid_planned_action')
        result=env.safe_step([21],reason='timeout')
        self.assertFalse(env.active); self.assertEqual(result[-1]['fallback_reason'],'timeout')


if __name__=='__main__': unittest.main()
