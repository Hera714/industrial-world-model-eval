"""总说明：验证四个工具声明、严格参数白名单以及错误码返回。"""
import unittest
from unittest.mock import patch
from agents.schemas import TOOL_DEFINITIONS,validate_arguments
from agents.tools import invoke

class ToolSchemaTest(unittest.TestCase):
    def test_four_json_schema_contracts(self):
        definitions={x['function']['name']:x['function']['parameters'] for x in TOOL_DEFINITIONS}
        self.assertEqual(set(definitions),{'list_envs','validate_config','run_benchmark','summarize_results'})
        for schema in definitions.values():
            self.assertEqual(schema['type'],'object');self.assertFalse(schema['additionalProperties'])
    def test_missing_and_extra_arguments(self):
        for name,args in [('validate_config',{}),('run_benchmark',{'config':{},'shell':'id'}),('list_envs',{'extra':1})]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):validate_arguments(name,args)
                with patch('agents.tools.run_benchmark') as runner:
                    result=invoke(name,args)
                    self.assertFalse(result['ok']);self.assertIsNotNone(result['error']['code']);runner.assert_not_called()
    def test_unknown_tool_rejected(self):
        self.assertFalse(invoke('shell',{})['ok'])

if __name__=='__main__':unittest.main()
