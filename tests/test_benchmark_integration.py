"""总说明：真实 BOPTEST 的短时端到端验收；显式启用，离线单测不会依赖服务。"""
import csv
import json
import math
import os
from pathlib import Path
import unittest

import numpy as np
from benchmark.runner import run_benchmark

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/benchmark/boptest_smoke.yaml'


@unittest.skipUnless(os.environ.get('BOPTEST_INTEGRATION') == '1', '设置 BOPTEST_INTEGRATION=1 启用真实服务测试')
class BoptestIntegrationTest(unittest.TestCase):
    def test_real_boptest_pipeline(self):
        """验证两个动力学控制器及内置基线；保留真实运行目录供人工检查。"""
        path = run_benchmark(CONFIG, ROOT)
        print('集成测试产物：' + str(path), flush=True)
        for name in ('config.yaml', 'versions.json', 'input_hashes.json', 'metrics.json',
                     'trajectories.csv', 'events.jsonl', 'run.log', 'status.json',
                     'predictions.npz', 'report.md', 'artifacts.json'):
            self.assertTrue((path / name).is_file(), name)
        status = json.loads((path / 'status.json').read_text())
        self.assertEqual(status['status'], 'completed')
        metrics = json.loads((path / 'metrics.json').read_text())['metrics']
        self.assertEqual({x['model'] for x in metrics['open_loop']}, {'persistence', 'arx'})
        self.assertEqual({x['model'] for x in metrics['closed_loop']}, {'persistence', 'arx', 'embedded'})
        with (path / 'trajectories.csv').open(encoding='utf-8', newline='') as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), status['trajectory_rows'])
        for model in ('persistence', 'arx', 'embedded'):
            rows = [json.loads(r['payload']) for r in records if r['task'] == 'closed_loop' and r['model'] == model]
            self.assertEqual(len(rows), 104)  # 96 步历史上下文 + 8 步评分。
            for index, row in enumerate(rows):
                self.assertEqual(row['step'], index)
                self.assertEqual(row['timestamp'], rows[0]['timestamp'] + 900 * index)
                self.assertEqual(row['context'], index < 96)
                self.assertFalse(row['terminated'])
                self.assertEqual(row['truncated'], index == 103)
                self.assertTrue(math.isfinite(row['energy_kwh']))
                self.assertGreaterEqual(row['energy_kwh'], 0)
                self.assertTrue(math.isfinite(row['applied_action_c']))
                if row['override_active']:
                    self.assertAlmostEqual(row['requested_action_c'], row['applied_action_c'], places=5)
                    self.assertTrue(18 <= row['applied_action_c'] <= 24)
                else:
                    self.assertIsNone(row['requested_action_c'])
            scored = rows[96:]
            summary = next(x for x in metrics['closed_loop'] if x['model'] == model)
            self.assertEqual(summary['steps'], 8)
            self.assertAlmostEqual(sum(x['energy_kwh'] for x in scored), summary['energy_kwh'], places=8)
            native_delta = (summary['native_final_kpi']['ener_tot'] - summary['native_context_kpi']['ener_tot']) * 192
            self.assertAlmostEqual(native_delta, summary['energy_kwh'], places=7)
            if model == 'embedded':
                self.assertTrue(all(not r['override_active'] for r in scored))
        with np.load(path / 'predictions.npz', allow_pickle=False) as arrays:
            self.assertTrue(any(key.endswith('__prediction') for key in arrays.files))
            self.assertTrue(any(key.endswith('__valid') for key in arrays.files))


if __name__ == '__main__':
    unittest.main()
