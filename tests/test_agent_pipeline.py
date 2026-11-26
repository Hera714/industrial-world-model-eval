"""总说明：模拟 LLM 和工具结果验证编排成功、校验拒绝和异常追溯。"""
import json,unittest
from unittest.mock import patch
from agents.pipeline import run_agent,ROOT
from agents.config_generator import config_from_intent

class PipelineTest(unittest.TestCase):
    def generator(self,*args,**kwargs):
        intent=dict(environment='boptest',models=['arx'])
        return dict(intent=intent,config=config_from_intent(intent),metadata=dict(model='mock',temperature=0))
    def test_success_uses_validated_tools_in_order(self):
        calls=[]
        def tool(name,args):
            calls.append(name)
            return dict(ok=True,result={'run_directory':str(ROOT/'reports/runs/test')} if name=='run_benchmark' else {'passed':True},error=None)
        with patch('agents.pipeline.invoke',side_effect=tool):
            path,state=run_agent('test',endpoint='mock',model='mock',generator=self.generator)
        self.assertEqual(calls,['validate_config','run_benchmark','summarize_results'])
        self.assertEqual(state['status'],'completed')
        self.assertEqual(json.loads((path/'session.json').read_text())['status'],'completed')
    def test_rejection_never_executes(self):
        def reject(name,args):return dict(ok=False,result=None,error=dict(message='bad config'))
        with patch('agents.pipeline.invoke',side_effect=reject) as tool:
            _,state=run_agent('test',endpoint='mock',model='mock',generator=self.generator)
        self.assertEqual(state['status'],'failed');self.assertEqual(tool.call_count,1)
    def test_large_request_cannot_skip_approval(self):
        def large_generator(*args,**kwargs):
            intent=dict(environment='boptest',models=['arx','persistence','mlp_seed42'])
            return dict(intent=intent,config=config_from_intent(intent),metadata={})
        with patch('agents.pipeline.invoke',return_value=dict(ok=True,result={},error=None)) as tool:
            _,state=run_agent('test',endpoint='mock',model='mock',generator=large_generator)
        self.assertEqual(state['status'],'failed')
        self.assertIn('APPROVAL_REQUIRED',state['error']['message'])
        self.assertEqual(tool.call_count,1)
    def test_llm_timeout_saved(self):
        def timeout(*args,**kwargs):raise TimeoutError('model timeout')
        path,state=run_agent('test',endpoint='mock',model='mock',generator=timeout)
        self.assertEqual(state['error']['type'],'TimeoutError')
        self.assertTrue((path/'session.json').exists())

if __name__=='__main__':unittest.main()
