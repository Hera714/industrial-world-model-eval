"""验证自主编排不能越过配置验证或改变用户指定的模型。"""
import unittest
from agents.react import run_react


class ReactGuardTest(unittest.TestCase):
    def test_run_without_validation_is_not_called(self):
        calls = []
        result = run_react('运行 BOPTEST arx', lambda r, h: {'tool': 'run_benchmark',
            'arguments': {'config': {'model': {'instances': [{'name': 'arx'}]}}}},
            max_turns=1, tool=lambda n, a: calls.append(n))
        self.assertEqual(calls, [])
        self.assertIn('VALIDATION_REQUIRED', result['history'][0]['observation']['error']['message'])

    def test_changed_config_cannot_run(self):
        configs = [{'model': {'instances': [{'name': 'arx'}]}},
                   {'model': {'instances': [{'name': 'persistence'}]}}]
        calls = []
        def decide(request, history):
            return {'tool': 'validate_config' if not history else 'run_benchmark',
                    'arguments': {'config': configs[bool(history)]}}
        def tool(name, args):
            calls.append(name)
            return {'ok': True}
        run_react('运行 BOPTEST arx', decide, max_turns=2, tool=tool)
        self.assertEqual(calls, ['validate_config'])

    def test_wrong_environment_cannot_run(self):
        config = {'model': {'instances': [{'name': 'arx'}]}}
        calls = []
        def decide(request, history):
            return {'tool': 'validate_config' if not history else 'run_benchmark', 'arguments': {'config': config}}
        def tool(name, args):
            calls.append(name)
            return {'ok': True}
        run_react('运行 IB arx', decide, max_turns=2, tool=tool)
        self.assertEqual(calls, ['validate_config'])

    def test_successful_configuration_is_not_run_twice(self):
        config = {'model': {'instances': [{'name': 'arx'}]}}
        calls = []
        def decide(request, history):
            return {'tool': 'validate_config' if not history else 'run_benchmark', 'arguments': {'config': config}}
        def tool(name, args):
            calls.append(name)
            return {'ok': True, 'result': {}}
        result = run_react('运行 BOPTEST arx', decide, max_turns=3, tool=tool)
        self.assertEqual(calls, ['validate_config', 'run_benchmark'])
        self.assertIn('DUPLICATE_RUN', result['history'][-1]['observation']['error']['message'])

    def test_old_run_cannot_be_used_as_current_evidence(self):
        calls = []
        result = run_react('运行 BOPTEST arx', lambda r, h: {'tool': 'summarize_results',
            'arguments': {'run_directory': '/some/old/run'}}, max_turns=1,
            tool=lambda n, a: calls.append(n))
        self.assertEqual(calls, [])
        self.assertIn('RUN_REQUIRED', result['history'][0]['observation']['error']['message'])

    def test_large_run_cannot_skip_approval(self):
        config={'model':{'instances':[{'name':'arx'},{'name':'persistence'},{'name':'mlp_seed42'}]}}
        calls=[]
        def decide(request,history):
            return {'tool':'validate_config' if not history else 'run_benchmark','arguments':{'config':config}}
        def tool(name,args):
            calls.append(name)
            return {'ok':True,'result':{}}
        result=run_react('BOPTEST arx persistence mlp_seed42',decide,max_turns=2,tool=tool)
        self.assertEqual(calls,['validate_config'])
        self.assertIn('APPROVAL_REQUIRED',result['history'][-1]['observation']['error']['message'])


if __name__ == '__main__':
    unittest.main()
