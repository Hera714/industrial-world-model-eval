"""总说明：离线核查结果完整性、损坏拒绝与子任务身份确定性。"""
import importlib.util
import json,tempfile,unittest,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('batch',ROOT/'scripts/batch_benchmark.py')
batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)

class BatchTest(unittest.TestCase):
    def test_result_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'status.json').write_text(json.dumps(dict(status='completed')))
            (path/'trajectory.csv').write_text('step\n0\n')
            original=batch.result_hashes(path)
            self.assertTrue(batch.verify_result(path,original))
            (path/'trajectory.csv').write_text('changed')
            self.assertFalse(batch.verify_result(path,original))
    def test_partial_and_missing_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            self.assertFalse(batch.verify_result(path,{}))
            (path/'status.json').write_text(json.dumps(dict(status='running')))
            self.assertFalse(batch.verify_result(path,batch.result_hashes(path)))
    def test_identity_changes_with_each_axis(self):
        identities=[dict(model=m,episode=e,condition=c,task='closed_loop') for m in ['arx','mlp'] for e in ['day1','day2'] for c in ['clean','noise']]
        keys={hashlib.sha256(json.dumps(i,sort_keys=True).encode()).hexdigest()[:24] for i in identities}
        self.assertEqual(len(keys),8)

if __name__=='__main__':unittest.main()
