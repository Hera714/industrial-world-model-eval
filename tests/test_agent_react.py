"""总说明：验证行动观察循环能处理配置错误、修正及防止伪造完成。"""
import unittest
from agents.react import run_react
class ReActTest(unittest.TestCase):
    def test_correction_after_failed_validation(self):
        def decide(request,history):
            config={'model':{'instances':[{'name':'arx'}]}}
            actions=[dict(tool='validate_config',arguments={'config':config,'bad':1}),dict(tool='validate_config',arguments={'config':config}),dict(tool='run_benchmark',arguments={'config':config}),dict(tool='summarize_results',arguments={'run_directory':'run-1'}),dict(tool='finish')]
            return actions[len(history)]
        def tool(name,args):return dict(ok='bad' not in args,result={'passed':True,'run_directory':'run-1'},error=None)
        result=run_react('运行 BOPTEST arx',decide,tool=tool)
        self.assertEqual(result['status'],'completed');self.assertFalse(result['history'][0]['observation']['ok'])
    def test_finish_without_audit_unverified(self):
        self.assertEqual(run_react('x',lambda r,h:dict(tool='finish'))['status'],'unverified')
    def test_unknown_tool_is_observation(self):
        result=run_react('x',lambda r,h:dict(tool='shell'),max_turns=1)
        self.assertEqual(result['status'],'turn_limit');self.assertFalse(result['history'][0]['observation']['ok'])
if __name__=='__main__':unittest.main()
