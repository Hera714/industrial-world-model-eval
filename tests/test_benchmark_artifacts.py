"""总说明：验证每次 run 独立、状态可追溯、失败保留前缀及原始非法数值。"""
from pathlib import Path
import copy
import csv
import hashlib
import json
import tempfile
import unittest
import numpy as np
from benchmark.config import load_config
from benchmark.artifacts import RunArtifacts

ROOT = Path(__file__).resolve().parents[1]


class ArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.config = load_config(ROOT/'configs/benchmark/boptest_smoke.yaml', ROOT)
        self.config['output']['directory'] = str(Path(self.temp.name).relative_to(ROOT))

    def test_completed_run_manifest(self):
        with RunArtifacts(self.config, ROOT) as run:
            run.set_context(task='open_loop', model='arx', episode='id_test_day160', step=0)
            run.append_trajectory(dict(episode='id_test_day160', step=0, prediction=[20., 21.]))
            run.write_metrics({'rmse': .2, 'failure_count': 0})
            run.write_predictions(predicted=np.ones((1, 6)), valid=np.ones(1, bool))
            run.write_report('# 验收样例\n这是产物模块测试，并非仿真结果。\n')
        state = json.loads((run.path/'status.json').read_text())
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['trajectory_rows'], 1)
        for name in ['config.yaml', 'versions.json', 'input_hashes.json', 'metrics.json',
                     'trajectories.csv', 'events.jsonl', 'run.log', 'context.json', 'report.md', 'predictions.npz']:
            self.assertTrue((run.path/name).is_file(), name)
        for name, spec in json.loads((run.path/'artifacts.json').read_text()).items():
            self.assertEqual(hashlib.sha256((run.path/name).read_bytes()).hexdigest(), spec['sha256'])
        with np.load(run.path/'predictions.npz', allow_pickle=False) as data:
            self.assertEqual(data['predicted'].shape, (1, 6))
        with self.assertRaisesRegex(RuntimeError, 'finalized'):
            run.write_metrics({'rmse': 0})

    def test_separate_directories_same_run_name(self):
        first, second = RunArtifacts(self.config, ROOT), RunArtifacts(self.config, ROOT)
        self.assertNotEqual(first.path, second.path)
        first.fail(RuntimeError('intentional test'))
        second.fail(RuntimeError('intentional test'))

    def test_exception_preserves_context_and_rows(self):
        with self.assertRaisesRegex(TimeoutError, 'advance timeout'):
            with RunArtifacts(self.config, ROOT) as run:
                run.set_context(task='closed_loop', episode='id_test_day160', step=3)
                run.append_trajectory({'step': 2, 'zone_c': 21.})
                raise TimeoutError('advance timeout')
        failure = json.loads((run.path/'failure.json').read_text())
        self.assertEqual(failure['error_type'], 'TimeoutError')
        self.assertEqual(failure['context']['step'], 3)
        self.assertIn('TimeoutError', failure['traceback'])
        self.assertEqual(json.loads((run.path/'status.json').read_text())['status'], 'failed')
        with (run.path/'trajectories.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_empty_run_cannot_be_completed(self):
        with self.assertRaisesRegex(RuntimeError, 'without metrics'):
            with RunArtifacts(self.config, ROOT) as run:
                pass
        self.assertEqual(json.loads((run.path/'status.json').read_text())['status'], 'failed')

    def test_column_mismatch_does_not_delete_prefix(self):
        with self.assertRaisesRegex(ValueError, 'columns mismatch'):
            with RunArtifacts(self.config, ROOT) as run:
                run.append_trajectory({'step': 0, 'zone_c': 21.})
                run.append_trajectory({'step': 1})
        self.assertEqual(json.loads((run.path/'failure.json').read_text())['trajectory_rows'], 1)

    def test_nonfinite_metrics_explicit_and_npz_original(self):
        with RunArtifacts(self.config, ROOT) as run:
            run.append_trajectory({'step': 0, 'predicted': float('nan')})
            run.write_metrics({'rmse': np.nan, 'failure_count': 1})
            run.write_predictions(predicted=np.array([np.nan, np.inf]))
        metrics = json.loads((run.path/'metrics.json').read_text())
        self.assertEqual(metrics['metrics']['rmse'], {'nonfinite': 'nan'})
        with np.load(run.path/'predictions.npz') as data:
            self.assertTrue(np.isnan(data['predicted'][0]))

    def test_invalid_config_creates_no_run(self):
        config = copy.deepcopy(self.config)
        config['env']['action']['api_unit'] = 'degC'
        with self.assertRaises(ValueError):
            RunArtifacts(config, ROOT)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
