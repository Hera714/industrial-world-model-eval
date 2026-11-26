"""总说明：验证模型拒绝、畸形 JSON 和非法模型均保留原始响应。"""
import unittest
from agents.response_parser import parse_response,ModelResponseError
from agents.errors import describe_error

class ResponseParserTest(unittest.TestCase):
    def test_errors_preserve_raw(self):
        for raw in ['not-json','{"error":"无法确定"}','{"environment":"boptest","models":["invalid"]}']:
            with self.subTest(raw=raw):
                with self.assertRaises(ModelResponseError) as context:parse_response(raw)
                self.assertEqual(context.exception.raw_response,raw)
                self.assertEqual(describe_error(context.exception)['raw_response'],raw)
    def test_valid_response(self):
        intent,config=parse_response('{"environment":"boptest","models":["arx"]}')
        self.assertEqual(intent['models'],['arx'])
        self.assertEqual(config['model']['instances'][0]['name'],'arx')

if __name__=='__main__':unittest.main()
