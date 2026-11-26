"""总说明：使用真实本地 MLflow 存储验证 Runner 成功和失败均可查询。"""
from pathlib import Path
import tempfile,json,unittest
import yaml
from benchmark.config import load_config
from benchmark.runner import run_benchmark
from tests.test_benchmark_runner import MemoryEnv

ROOT=Path(__file__).resolve().parents[1]

class TrackingTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.cfg=load_config(ROOT/'configs/benchmark/boptest_smoke.yaml',ROOT)
        self.cfg['output']['directory']=str((self.folder/'runs').relative_to(ROOT))
        self.tracking=self.folder/'tracking.yaml'
        self.tracking.write_text(yaml.safe_dump(dict(backend='local_file',directory=str((self.folder/'mlruns').relative_to(ROOT)),experiment='test')))
        MemoryEnv.instances=[];MemoryEnv.fail_step=None

    def query(self,path):
        from mlflow.tracking import MlflowClient
        ref=json.loads((path/'tracking.json').read_text())
        return MlflowClient(tracking_uri=ref['tracking_uri']).get_run(ref['mlflow_run_id'])

    def test_finished_parameters_hashes_metrics_and_path(self):
        path=run_benchmark(self.cfg,ROOT,MemoryEnv,tracking_config=self.tracking)
        run=self.query(path)
        self.assertEqual(run.info.status,'FINISHED')
        self.assertEqual(run.data.tags['benchmark.directory'],str(path))
        self.assertEqual(run.data.params['config.model.instances.0.checkpoint.sha256'],json.dumps(self.cfg['model']['instances'][0]['checkpoint']['sha256']))
        self.assertIn('closed_loop.0.cost',run.data.metrics)
        self.assertIn('open_loop.0.rmse.zone_c',run.data.metrics)

    def test_failed_run_query_retains_context(self):
        MemoryEnv.fail_step=3
        with self.assertRaises(TimeoutError):
            run_benchmark(self.cfg,ROOT,MemoryEnv,tracking_config=self.tracking)
        path=next((self.folder/'runs').iterdir())
        run=self.query(path)
        self.assertEqual(run.info.status,'FAILED')
        self.assertEqual(run.data.tags['benchmark.error_type'],'TimeoutError')
        self.assertEqual(json.loads((path/'failure.json').read_text())['context']['step'],3)

if __name__=='__main__':unittest.main()
