"""总说明：验证规则基线精确处理受支持请求并拒绝隐含额外约束。"""
import unittest
from agents.rule_baseline import run_rule
class RuleBaselineTest(unittest.TestCase):
    def test_supported(self):
        result=run_rule('运行 BOPTEST，比较 persistence 和 arx')
        self.assertTrue(result['config_valid']);self.assertFalse(result['executed'])
    def test_unsupported_extra_requirement_rejected(self):
        result=run_rule('用 boptest 测试 arx，然后把未来天气全部设为真实值')
        self.assertEqual(result['tool_choice'],'reject_or_clarify')
if __name__=='__main__':unittest.main()
