"""总说明：人为触发MPC超时、无有效候选和规划异常，端到端检查释放命令与实际执行信息。"""
import itertools
import unittest
from unittest.mock import patch
import numpy as np
from controllers.boptest_mpc import BoptestShootingMPC
from tests.test_boptest_control import FakeControl


class FaultModel:
    """故障只作用于规划模型，环境仍按正常推进接口执行后备动作。"""
    def __init__(self, fault):
        self.fault = fault

    def predict(self, history):
        if self.fault == 'exception':
            raise RuntimeError('Injected prediction failure')
        result = np.column_stack([history[:, -1, :5], np.full(len(history), .2)])
        if self.fault == 'negative_energy':
            result[:, 5] = -1.
        elif self.fault == 'partial_invalid':
            result[0, 5] = -1.
        return result


def plan_fault(fault):
    """保留正式候选数/时域，计时故障采用确定性时钟，避免依赖机器负载。"""
    config = dict(action=dict(experiment_bounds_c=[18., 24.]), control=dict(
        horizon=12, candidates=256, action_hold_steps=4, time_budget_s=5.,
        comfort_weight=1., energy_weight=1., action_change_weight=.02))
    history = np.ones((16, 12))
    history[:, :3] = [21., 35., 30.]
    controller = BoptestShootingMPC(FaultModel(fault), config)
    arguments = (history, 86400, np.full(12, 21.), np.full(12, 24.), 96, 22.)
    if fault == 'timeout':
        clock = itertools.count(0., 6.)
        with patch('controllers.boptest_mpc.time.perf_counter', side_effect=lambda: next(clock)):
            return controller.act(*arguments)
    return controller.act(*arguments)


class Week6FallbackTest(unittest.TestCase):
    def check_release(self, fault, expected_reason):
        env = FakeControl()
        env.step([23.])
        self.assertTrue(env.active)
        action, detail = plan_fault(fault)
        self.assertIsNone(action)
        self.assertEqual(detail['fallback_reason'], expected_reason)
        result = env.safe_step(action, detail['fallback_reason'])
        self.assertEqual(env.commands[-1], {'oveTSet_activate': 0})
        self.assertFalse(env.active)
        self.assertFalse(result[-1]['override_active'])
        self.assertIsNone(result[-1]['requested_action_c'])
        self.assertAlmostEqual(result[-1]['applied_action_c'], 21.)
        self.assertEqual(result[-1]['fallback_reason'], expected_reason)
        self.assertEqual(env.steps, 2)
        self.assertTrue(result[3])
        return detail

    def test_timeout_releases_prior_override(self):
        detail = self.check_release('timeout', 'timeout')
        self.assertGreaterEqual(detail['decision_time_s'], 5.)

    def test_all_candidates_invalid_releases(self):
        detail = self.check_release('negative_energy', 'no_valid_candidates')
        self.assertEqual(detail['valid_candidates'], 0)

    def test_prediction_exception_releases_and_preserves_error(self):
        detail = self.check_release('exception', 'planning_error')
        self.assertIn('Injected prediction failure', detail['error'])

    def test_partial_invalid_candidates_keep_valid_action(self):
        action, detail = plan_fault('partial_invalid')
        self.assertEqual(detail['fallback_reason'], '')
        self.assertGreater(detail['valid_candidates'], 0)
        self.assertTrue(18 <= action[0] <= 24)
        env = FakeControl()
        result = env.safe_step(action)
        self.assertTrue(result[-1]['override_active'])


if __name__ == '__main__':
    unittest.main()
