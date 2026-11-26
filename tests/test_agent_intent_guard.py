"""总说明：验证明确环境约束，防止其他模拟器被模型悄悄替换为 BOPTEST。"""
import unittest
from agents.intent_guard import require_explicit_boptest

class IntentGuardTest(unittest.TestCase):
    def test_explicit_boptest(self):require_explicit_boptest('在 BOPTEST 中比较 arx 和 persistence')
    def test_other_environments_and_ambiguous_rejected(self):
        for request in ['在 IB 环境运行 arx','在 EnergyPlus 中运行 arx','比较 BOPTEST 和 CityLearn','运行 arx']:
            with self.subTest(request=request):
                with self.assertRaisesRegex(ValueError,'ENV_NOT_ALLOWED'):require_explicit_boptest(request)

if __name__=='__main__':unittest.main()
