"""总说明：用恶意与错误模型输出测试冻结测试集的真实执行闸门。"""
import unittest
from agents.evaluation_guard import may_execute

class EvaluationGuardTest(unittest.TestCase):
    def test_permits_only_exact_authorized_prediction(self):
        case=dict(expected_action='generate_and_validate',expected_models=['arx'])
        self.assertTrue(may_execute(case,{'models':['arx']},True))
        self.assertFalse(may_execute(case,{'models':['persistence']},True))
        self.assertFalse(may_execute(case,{'models':['arx']},False))
    def test_invalid_instruction_never_executes(self):
        case=dict(expected_action='reject_or_clarify',expected_models=[])
        self.assertFalse(may_execute(case,{'models':['arx']},True))

if __name__=='__main__':unittest.main()
