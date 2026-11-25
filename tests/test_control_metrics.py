# 总说明：用人工计算的轨迹核验统一成本、能耗、约束率和后备计数。
import unittest
from metrics.control import calculate_control_metrics
class ControlMetricTest(unittest.TestCase):
    def test_hand_calculation(self):
        rows=[dict(true_air_after_c=20,target_c=21,action_kw=2,decision_time_s=.001,comfort_violation=False,fallback_reason=''),
              dict(true_air_after_c=23,target_c=21,action_kw=4,decision_time_s=.003,comfort_violation=True,fallback_reason='timeout')]
        m=calculate_control_metrics(rows,.5)
        self.assertAlmostEqual(m['tracking_mae_c'],1.5); self.assertAlmostEqual(m['energy_kwh'],3)
        self.assertAlmostEqual(m['cumulative_cost'],8); self.assertEqual(m['comfort_violation_rate'],.5)
        self.assertAlmostEqual(m['decision_mean_ms'],2); self.assertEqual(m['fallback_reasons'],{'timeout':1})
    def test_invalid(self):
        with self.assertRaises(ValueError): calculate_control_metrics([],1)
if __name__=='__main__': unittest.main()
