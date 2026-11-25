"""有效性测试：手工触发非有限、越界、速率异常与模型异常，验证保留前缀。"""
import unittest
from dataclasses import asdict
import numpy as np
from metrics.validity import check_prediction, thermal_rate_bounds
from simulators.simple_thermal import ThermalConfig
from models.dynamics_base import DynamicsModel, PredictionOutputError
from evaluators.rollout import rollout_episode


CFG = dict(observation_bounds_c=[None,None], plausibility_bounds_c=[-10,50],
           divergence_bounds_c=[-100,100], rate_check=dict(outdoor_bounds_c=[-5,17],
           power_bounds_kw=[0,5], load_kw=.3, noise_margin_c=.03))


class FailingDynamics(DynamicsModel):
    """第二次状态递推触发指定故障，不依赖调用次数，兼容批量重试。"""
    def __init__(self, fault):
        super().__init__()
        self.fault = fault

    def _predict(self, state, action, exogenous):
        if (state[..., 0] >= 1).any():
            if self.fault == 'exception':
                raise RuntimeError('injected failure')
            if self.fault == 'shape':
                return np.zeros((1,))
            return np.full_like(state, self.fault)
        return state+1


class ValidityTest(unittest.TestCase):
    def test_thresholds_and_unbounded_observation_space(self):
        result = check_prediction(np.array([[20.,19.]]), np.array([[51.,19.]]), CFG, asdict(ThermalConfig()))
        self.assertTrue(result['events']['plausibility_bounds'][0,0])
        self.assertTrue(result['events']['rate'][0,0])
        self.assertFalse(result['events']['observation_bounds'].any())
        self.assertFalse(result['stop'][0])
        self.assertEqual(result['details']['plausibility_bounds']['magnitude'][0,0], 1)
        result = check_prediction(np.array([[20.,19.]]), np.array([[101.,19.]]), CFG)
        self.assertTrue(result['events']['divergence'][0,0])
        self.assertTrue(result['stop'][0])

    def test_rate_bounds_contain_physical_extremes(self):
        physics = asdict(ThermalConfig())
        x = np.array([[20.,19.]])
        low, high = thermal_rate_bounds(x, CFG['rate_check'], physics)
        self.assertTrue((low < high).all())
        for value in (x+low, x+high):
            self.assertFalse(check_prediction(x, value, CFG, physics)['events']['rate'].any())

    def test_nonfinite_raw_and_prefix(self):
        for fault, event in [(np.nan,'nan'), (np.inf,'inf')]:
            with self.subTest(event=event):
                model = FailingDynamics(fault)
                result = rollout_episode(model, np.zeros((4,2)), np.zeros((4,1)), np.zeros((4,2)),
                    np.zeros((4,2)), 4, 'free_running', 'recorded_observation', validity_config=CFG)
                self.assertEqual(result['failure_step'][0], 2)
                np.testing.assert_equal(result['predicted'][0,0], [1,1])
                self.assertTrue(np.isnan(result['predicted'][0,1:]).all())
                np.testing.assert_equal(result['attempted'][0], [True,True,False,False])
                self.assertTrue(any(row['event'] == event and row['horizon'] == 2 for row in result['events']))
                if event == 'inf':
                    self.assertTrue(np.isinf(result['raw'][0,1]).all())

    def test_shape_and_inference_errors(self):
        for fault in ('shape','exception'):
            result = rollout_episode(FailingDynamics(fault), np.zeros((3,2)), np.zeros((3,1)),
                np.zeros((3,2)), np.zeros((3,2)), 3, 'free_running', 'recorded_observation', validity_config=CFG)
            self.assertEqual(result['failure_step'][0], 2)
            self.assertEqual(result['events'][0]['event'], 'inference_error')
            self.assertFalse(any(row['event'] == 'nan' for row in result['events']))


if __name__ == '__main__':
    unittest.main()
