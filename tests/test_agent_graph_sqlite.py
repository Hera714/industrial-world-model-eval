"""总说明：真实 SQLite 检查点在关闭连接和重新构建状态机后恢复人工审批。"""
import sqlite3,tempfile,unittest
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from agents.graph import build_graph

def generator(*args,**kwargs):return dict(config={'example':1},metadata={},intent={})
def tool(name,args):
    return dict(ok=True,result={'run_directory':'/example'} if name=='run_benchmark' else {'passed':True},error=None)

class SQLiteGraphTest(unittest.TestCase):
    def test_resume_after_connection_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'checkpoints.sqlite';config={'configurable':{'thread_id':'persisted'}}
            with sqlite3.connect(file,check_same_thread=False) as connection:
                graph=build_graph(generator,tool,SqliteSaver(connection))
                state=graph.invoke(dict(request='x',endpoint='x',model='x',require_approval=True),config)
                self.assertIn('__interrupt__',state)
            with sqlite3.connect(file,check_same_thread=False) as connection:
                graph=build_graph(generator,tool,SqliteSaver(connection))
                result=graph.invoke(Command(resume={'approved':True,'config_sha256':state['__interrupt__'][0].value['config_sha256']}),config)
                self.assertEqual(result['status'],'completed')
                self.assertEqual([e['node'] for e in result['events']],['generate','validate','approval','execute','audit','summary'])

if __name__=='__main__':unittest.main()
