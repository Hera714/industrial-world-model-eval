"""总说明：验证评测分母、缺失结果及重复身份处理。"""
import unittest
from evaluators.agent import score_cases

class AgentEvaluationTest(unittest.TestCase):
    def test_denominators_and_missing(self):
        cases=[dict(id='a',category='valid',expected_action='generate_and_validate',expected_models=['arx']),
               dict(id='b',category='invalid',expected_action='reject_or_clarify',expected_models=[])]
        out=[dict(id='a',config_valid=True,models=['arx'],tool_choice='validate_config',executed=True,audit_passed=True,execution_evaluated=True)]
        result=score_cases(cases,out)
        self.assertEqual(result['summary']['tool_correct'],dict(numerator=1,denominator=2,rate=.5))
        self.assertEqual(result['summary']['error_identified']['denominator'],1)
        self.assertTrue(result['cases'][1]['missing'])
    def test_reject_duplicate_or_unknown(self):
        case=dict(id='a',category='valid',expected_action='generate_and_validate',expected_models=[])
        with self.assertRaises(ValueError):score_cases([case,case],[])
        with self.assertRaises(ValueError):score_cases([case],[dict(id='b')])

if __name__=='__main__':unittest.main()
