"""总说明：篡改指标后同步文件哈希，确认审核仍发现单位和原生 KPI 不一致。"""
import hashlib,json,shutil,tempfile,unittest
from pathlib import Path
from agents.result_auditor import audit_run
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/runs/20261125T122045898111Z_boptest_smoke_container_fb64e772392f'

class KPIAuditTest(unittest.TestCase):
    def setUp(self):
        if not SOURCE.exists():self.skipTest('真实 smoke 结果不可用')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'run';shutil.copytree(SOURCE,self.path)
    def mutate(self,edit):
        file=self.path/'metrics.json';data=json.loads(file.read_text());edit(data['metrics']);file.write_text(json.dumps(data))
        manifest_file=self.path/'artifacts.json';manifest=json.loads(manifest_file.read_text())
        manifest['metrics.json']['sha256']=hashlib.sha256(file.read_bytes()).hexdigest();manifest_file.write_text(json.dumps(manifest))
    def test_bad_energy_conversion(self):
        self.mutate(lambda data:data['closed_loop'][0].update(energy_kwh=999))
        audit=audit_run(self.path)
        self.assertFalse(audit['passed']);self.assertTrue(any(x['name'].startswith('native_energy_conversion:') and not x['passed'] for x in audit['checks']))
    def test_bad_unit(self):
        self.mutate(lambda data:data['units'].__setitem__(0,'K'))
        audit=audit_run(self.path)
        self.assertFalse(audit['passed']);self.assertTrue(any(x['name']=='variable_units' and not x['passed'] for x in audit['checks']))

if __name__=='__main__':unittest.main()
