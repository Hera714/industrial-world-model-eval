"""总说明：使用已验收真实 smoke 的临时副本检验回归规则和负面案例，不需服务。"""
import copy
import json
import shutil
import tempfile
from pathlib import Path
import unittest
import yaml
from benchmark.regression import check_run

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/runs/20261125T120406326749Z_boptest_smoke_container_75e98fc068c2'


class RegressionTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'run'
        if not SOURCE.is_dir(): self.skipTest('本地真实 smoke 结果不存在；通过 CLI 对新 run 执行回归')
        shutil.copytree(SOURCE,self.path)
        self.policy=yaml.safe_load((ROOT/'configs/regression/boptest_smoke.yaml').read_text())

    def mutate_metrics(self, edit):
        import hashlib
        path=self.path/'metrics.json'
        data=json.loads(path.read_text());edit(data['metrics']);path.write_text(json.dumps(data))
        manifest_path=self.path/'artifacts.json';manifest=json.loads(manifest_path.read_text())
        manifest['metrics.json']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    def test_real_result_passes(self): self.assertEqual(check_run(self.path,self.policy),[])
    def test_missing_file_fails(self):
        (self.path/'report.md').unlink()
        self.assertTrue(any('missing files' in x for x in check_run(self.path,self.policy)))
    def test_failed_predictions_detected(self):
        def mutate(m):
            m['open_loop'][0].update(valid=0,failed=5,valid_fraction=0.)
        self.mutate_metrics(mutate)
        self.assertTrue(any('fraction' in x for x in check_run(self.path,self.policy)))
    def test_excess_fallback_detected(self):
        self.mutate_metrics(lambda m:m['closed_loop'][0].update(fallback_count=8))
        self.assertTrue(any('fallback_count' in x for x in check_run(self.path,self.policy)))
    def test_missing_model_and_nonfinite_detected(self):
        self.mutate_metrics(lambda m:(m['open_loop'].pop(),m['closed_loop'][0].update(cost=float('nan'))))
        errors=check_run(self.path,self.policy)
        self.assertTrue(any('matrix' in x for x in errors))
        self.assertTrue(any('cost' in x for x in errors))
    def test_changed_file_hash_detected(self):
        (self.path/'report.md').write_text('changed')
        self.assertTrue(any('hash mismatch' in x for x in check_run(self.path,self.policy)))
    def test_elapsed_limit(self):
        policy=copy.deepcopy(self.policy);policy['max_elapsed_seconds']=0
        self.assertTrue(any('elapsed' in x for x in check_run(self.path,policy)))


if __name__=='__main__': unittest.main()
