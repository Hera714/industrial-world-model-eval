"""总说明：验证 LangGraph 审批中断、恢复、拒绝及错误转移。"""
import unittest
from unittest.mock import patch
from pathlib import Path
from langgraph.types import Command
from agents.graph import build_graph

def generator(*args,**kwargs):return dict(config={'example':1},metadata={},intent={})
def tool(name,args):
    result={'run_benchmark':{'run_directory':'/test'},'summarize_results':{'passed':True}}.get(name,{})
    return dict(ok=True,result=result,error=None)

class GraphTest(unittest.TestCase):
    def test_approval_and_resume(self):
        graph=build_graph(generator,tool);config={'configurable':{'thread_id':'approve'}}
        state=graph.invoke(dict(request='x',endpoint='x',model='x',require_approval=True),config)
        self.assertIn('__interrupt__',state)
        result=graph.invoke(Command(resume={'approved':True,'config_sha256':state['__interrupt__'][0].value['config_sha256']}),config)
        self.assertEqual(result['status'],'completed')
        self.assertEqual([e['node'] for e in result['events']],['generate','validate','approval','execute','audit','summary'])
    def test_reject(self):
        graph=build_graph(generator,tool);config={'configurable':{'thread_id':'reject'}}
        graph.invoke(dict(request='x',endpoint='x',model='x',require_approval=True),config)
        self.assertEqual(graph.invoke(Command(resume={'approved':False}),config)['status'],'rejected')
    def test_mismatched_hash_rejected(self):
        calls=[]
        def tracked(name,args):
            calls.append(name)
            return tool(name,args)
        graph=build_graph(generator,tracked);config={'configurable':{'thread_id':'wrong-hash'}}
        graph.invoke(dict(request='x',endpoint='x',model='x',require_approval=True),config)
        result=graph.invoke(Command(resume={'approved':True,'config_sha256':'0'*64}),config)
        self.assertEqual(result['status'],'rejected')
        self.assertNotIn('run_benchmark',calls)

    def test_invalid_configuration_stops_before_execution(self):
        def bad(name,args):return dict(ok=False,error={'message':'bad config'},result=None)
        graph=build_graph(generator,bad)
        result=graph.invoke(dict(request='x',endpoint='x',model='x'),{'configurable':{'thread_id':'failure'}})
        self.assertEqual(result['status'],'failed')
        self.assertEqual([e['node'] for e in result['events']],['generate','validate'])

    def test_large_model_selection_requires_approval_automatically(self):
        calls=[]
        def many_models(*args,**kwargs):
            return dict(config={'model':{'instances':[{'name':str(i)} for i in range(3)]}},metadata={},intent={})
        def tracked(name,args):
            calls.append(name)
            return tool(name,args)
        graph=build_graph(many_models,tracked)
        state=graph.invoke(dict(request='x',endpoint='x',model='x',require_approval=False),
                           {'configurable':{'thread_id':'automatic-approval'}})
        self.assertIn('__interrupt__',state)
        self.assertEqual(calls,['validate_config'])

    def test_approved_large_graph_reaches_guarded_tool(self):
        config_data={'model':{'instances':[{'name':str(i)} for i in range(3)]}}
        def many(*args,**kwargs):return dict(config=config_data,metadata={},intent={})
        root=Path(__file__).resolve().parents[1]
        with patch('agents.tools.validate_candidate',return_value=config_data), \
             patch('agents.tools.run_benchmark',return_value=root/'reports/runs/mock') as runner, \
             patch('agents.tools.audit_run',return_value={'passed':True,'checks':[]}):
            graph=build_graph(generator=many)
            thread={'configurable':{'thread_id':'guarded-real-tool'}}
            state=graph.invoke(dict(request='x',endpoint='x',model='x'),thread)
            self.assertIn('__interrupt__',state)
            runner.assert_not_called()
            digest=state['__interrupt__'][0].value['config_sha256']
            result=graph.invoke(Command(resume={'approved':True,'config_sha256':digest}),thread)
            self.assertEqual(result['status'],'completed')
            runner.assert_called_once()

if __name__=='__main__':unittest.main()
