"""总说明：核查工具安全边界、模型白名单和审核失败案例。"""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from agents.tools import invoke
from agents.config_generator import config_from_intent
from agents.result_auditor import audit_run
ROOT=Path(__file__).resolve().parents[1]

class AgentToolsTest(unittest.TestCase):
    def test_allowed_model_selection(self):
        config=config_from_intent(dict(environment='boptest',models=['persistence','arx']))
        self.assertTrue(invoke('validate_config',{'config':config})['ok'])
    def test_invalid_intents_and_shell_rejected(self):
        for intent in [dict(environment='other',models=['arx']),dict(environment='boptest',models=['unknown']),dict(environment='boptest',models=['arx'],shell='id')]:
            with self.assertRaises(ValueError):config_from_intent(intent)
        with patch('agents.tools.run_benchmark') as runner:
            result=invoke('run_benchmark',{'config':{},'shell':'id'})
            self.assertFalse(result['ok']);runner.assert_not_called()
    def test_modified_url_or_checkpoint_rejected(self):
        cfg=config_from_intent(dict(environment='boptest',models=['arx']))
        for field in ('url','checkpoint'):
            changed=copy.deepcopy(cfg)
            if field=='url':changed['env']['url']='http://evil.invalid'
            else:changed['model']['instances'][0]['checkpoint']['sha256']='0'*64
            with patch('agents.tools.run_benchmark') as runner:
                self.assertFalse(invoke('run_benchmark',{'config':changed})['ok']);runner.assert_not_called()
    def test_large_run_requires_bound_approval(self):
        cfg=config_from_intent(dict(environment='boptest',models=['arx','persistence','mlp_seed42']))
        digest=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        with patch('agents.tools.run_benchmark',return_value=ROOT/'reports/runs/mock') as runner:
            self.assertFalse(invoke('run_benchmark',{'config':cfg})['ok'])
            runner.assert_not_called()
            self.assertFalse(invoke('run_benchmark',{'config':cfg},approved_config_hash='0'*64)['ok'])
            runner.assert_not_called()
            self.assertTrue(invoke('run_benchmark',{'config':cfg},approved_config_hash=digest)['ok'])
            runner.assert_called_once()
    def test_missing_results_fail_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            result=audit_run(directory)
            self.assertFalse(result['passed'])
            self.assertTrue(any(not c['passed'] for c in result['checks']))
    def test_existing_result_and_hash_tamper(self):
        import shutil
        source=ROOT/'reports/runs/20261125T122045898111Z_boptest_smoke_container_fb64e772392f'
        if not source.exists():self.skipTest('本地历史 run 不存在')
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'run';shutil.copytree(source,target)
            self.assertTrue(audit_run(target)['passed'])
            (target/'metrics.json').write_text('{}')
            self.assertFalse(audit_run(target)['passed'])

if __name__=='__main__':unittest.main()
