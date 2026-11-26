"""总说明：验证四类错误及其修复建议，不依赖模型服务。"""
import unittest
from agents.errors import describe_error
class ErrorTest(unittest.TestCase):
    def test_inf_requires_a_separate_numeric_token(self):
        for message in ('inference service unavailable','information missing'):
            self.assertEqual(describe_error(RuntimeError(message))['code'],'EXECUTION_ERROR')
        for message in ('prediction Inf','value -inf','NaN in output'):
            self.assertEqual(describe_error(ValueError(message))['code'],'INVALID_PREDICTION')
    def test_four_failure_types(self):
        cases=[(KeyError('model'),'MISSING_FIELD'),(ValueError('ENV_NOT_ALLOWED'),'INVALID_ENVIRONMENT'),
               (ValueError('invalid predicted state NaN'),'INVALID_PREDICTION'),(TimeoutError('timed out'),'TIMEOUT')]
        for error,expected in cases:
            with self.subTest(expected=expected):
                result=describe_error(error)
                self.assertEqual(result['code'],expected)
                self.assertTrue(result['message']);self.assertTrue(result['suggestion'])
if __name__=='__main__':unittest.main()
