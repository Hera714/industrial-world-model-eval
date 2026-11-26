"""总说明：注入计数一致的失败预测，验证审核不会因文件哈希合法而误报通过。"""
import hashlib,json,shutil,tempfile,unittest
from pathlib import Path
from agents.result_auditor import audit_run
ROOT=Path(__file__).resolve().parents[1]

class AuditorNegativeTest(unittest.TestCase):
    def test_failed_prediction_even_when_hashes_match(self):
        source=ROOT/'reports/runs/20261125T122045898111Z_boptest_smoke_container_fb64e772392f'
        if not source.exists():self.skipTest('本地验收 run 不存在')
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'run';shutil.copytree(source,target)
            metrics_path=target/'metrics.json';metrics=json.loads(metrics_path.read_text())
            row=metrics['metrics']['open_loop'][0];row.update(valid=4,failed=1,valid_fraction=.8)
            metrics_path.write_text(json.dumps(metrics))
            manifest_path=target/'artifacts.json';manifest=json.loads(manifest_path.read_text())
            manifest['metrics.json']['sha256']=hashlib.sha256(metrics_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            audit=audit_run(target)
            self.assertFalse(audit['passed'])
            self.assertTrue(any(c['name'].startswith('prediction_all_valid:') and not c['passed'] for c in audit['checks']))

if __name__=='__main__':unittest.main()
