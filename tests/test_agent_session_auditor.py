"""总说明：核验 Agent 会话对不一致配置、工具顺序和失败状态的检测。"""
import json,shutil,tempfile,unittest
from pathlib import Path
from agents.session_auditor import audit_session

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/week8/sessions/20261125T161119997809Z_c54402617b'

class SessionAuditTest(unittest.TestCase):
    def setUp(self):
        if not SOURCE.exists():self.skipTest('本地真实演示不存在')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)/'session';shutil.copytree(SOURCE,self.folder)
    def test_existing_session(self):self.assertTrue(audit_session(self.folder)['passed'])
    def test_modified_config_detected(self):
        path=self.folder/'config.yaml';path.write_text(path.read_text().replace('episode_steps: 8','episode_steps: 7'))
        result=audit_session(self.folder)
        self.assertFalse(result['passed']);self.assertTrue(any(x['name']=='same_config' and not x['passed'] for x in result['checks']))
    def test_missing_tool_and_failed_status(self):
        path=self.folder/'session.json';state=json.loads(path.read_text());state['events'].pop(0);state['status']='failed';path.write_text(json.dumps(state))
        result=audit_session(self.folder)
        self.assertFalse(result['passed']);self.assertTrue(any(x['name']=='tool_order' and not x['passed'] for x in result['checks']))

    def test_workspace_run_is_portable(self):
        path=self.folder/'session.json';state=json.loads(path.read_text())
        event=next(x for x in state['events'] if x['tool']=='run_benchmark')
        name=Path(event['result']['result']['run_directory']).name
        event['result']['result']['run_directory']='/workspace/reports/runs/'+name
        path.write_text(json.dumps(state))
        result=audit_session(self.folder)
        self.assertTrue(result['passed'],result)

    def test_relative_path_cannot_escape(self):
        path=self.folder/'session.json';state=json.loads(path.read_text())
        state['run_directory_relative']='reports/runs/../../outside'
        path.write_text(json.dumps(state))
        self.assertFalse(audit_session(self.folder)['passed'])

if __name__=='__main__':unittest.main()
