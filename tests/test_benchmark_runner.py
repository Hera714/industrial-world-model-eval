"""总说明：短回合运行真实加载器、MPC 和指标，仅用内存假服务替换 HTTP，另有独立真实服务验收。"""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from benchmark.config import load_config
from benchmark.runner import run_benchmark
from evaluators.benchmark_boptest import ConfiguredMPC

ROOT = Path(__file__).resolve().parents[1]


class MemoryEnv:
    """测试专用确定性环境；不当作 BOPTEST 真实物理结果。"""
    instances = []
    fail_step = None

    def __init__(self, config):
        self.config = config
        self.testid = 'test'
        self.steps = 0
        self.last_energy = 0.
        self.closed = False
        self.instances.append(self)

    def points(self):
        return dict(time=self.config['start_time'] + self.steps * 900,
                    reaTZon_y=294.15, reaTSup_y=303.15, reaTRet_y=298.15,
                    reaQFloHea_y=1000., reaPHeaPum_y=1000., reaPFan_y=0., reaPPumEmi_y=0.,
                    weaSta_reaWeaTDryBul_y=283.15, weaSta_reaWeaHGloHor_y=0.)

    def request(self, method, path, data=None):
        if path == '/version':
            return {'version': '0.9.0'}
        if path.startswith('/forecast/'):
            start = self.config['start_time']
            return {'time': [start, start + data['horizon']],
                    'LowerSetp[1]': [293.65, 293.65], 'UpperSetp[1]': [297.15, 297.15]}
        raise AssertionError('Unexpected request '+path)

    def reset(self, **kwargs):
        return np.array([294.15]), {'measurements': self.points()}

    def step(self, action):
        if self.fail_step is not None and self.steps == self.fail_step:
            raise TimeoutError('Injected advance timeout')
        self.steps += 1
        self.last_energy += .25
        return np.array([294.15]), 0., False, self.steps == self.config['episode_steps'], dict(
            measurements=self.points(), applied_action_c=21. if action is None else float(action[0]),
            override_active=action is not None, energy_kwh=.25, native_kpi={'ener_tot': self.last_energy/192}, fallback_reason='')

    def safe_step(self, action, reason=''):
        result = self.step(None if reason else action)
        result[-1]['fallback_reason'] = reason
        return result

    def close(self):
        self.closed = True


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.config = load_config(ROOT/'configs/benchmark/boptest_smoke.yaml', ROOT)
        self.config['output']['directory'] = str(Path(self.temp.name).relative_to(ROOT))
        MemoryEnv.instances = []
        MemoryEnv.fail_step = None

    def test_complete_pipeline_two_models(self):
        path = run_benchmark(self.config, ROOT, MemoryEnv)
        self.assertEqual(json.loads((path/'status.json').read_text())['status'], 'completed')
        metrics = json.loads((path/'metrics.json').read_text())['metrics']
        self.assertEqual({m['model'] for m in metrics['open_loop']}, {'arx', 'persistence'})
        self.assertEqual({m['model'] for m in metrics['closed_loop']}, {'arx', 'persistence', 'embedded'})
        self.assertTrue(all(m['steps'] == 8 for m in metrics['closed_loop']))
        self.assertTrue(all(m['requested'] == 5 for m in metrics['open_loop']))
        self.assertEqual(len(MemoryEnv.instances), 4)
        self.assertTrue(all(env.closed for env in MemoryEnv.instances))
        self.assertTrue((path/'report.md').is_file())
        self.assertTrue((path/'predictions.npz').is_file())

    def test_advance_failure_preserves_prefix_and_closes(self):
        MemoryEnv.fail_step = 3
        with self.assertRaises(TimeoutError):
            run_benchmark(self.config, ROOT, MemoryEnv)
        path = next(Path(self.temp.name).iterdir())
        failure = json.loads((path/'failure.json').read_text())
        self.assertEqual(failure['trajectory_rows'], 3)
        self.assertEqual(failure['context']['step'], 3)
        self.assertEqual(failure['context']['episode'], 'id_test_day160')
        self.assertEqual(json.loads((path/'status.json').read_text())['status'], 'failed')
        self.assertTrue(MemoryEnv.instances[0].closed)
        self.assertEqual(MemoryEnv.instances[0].steps, 3)

    def test_invalid_config_never_creates_environment(self):
        self.config['model']['input_features'] = 13
        with self.assertRaises(ValueError):
            run_benchmark(self.config, ROOT, MemoryEnv)
        self.assertEqual(MemoryEnv.instances, [])

    def test_recorded_open_loop_never_connects(self):
        self.config['dataset']['mode'] = 'recorded'
        self.config['evaluator']['tasks'] = ['open_loop']
        self.config['evaluator']['condition_matrix'].pop('closed_loop')
        with patch('socket.socket.connect', side_effect=AssertionError('No HTTP')):
            path = run_benchmark(self.config, ROOT, MemoryEnv)
        self.assertTrue((path/'report.md').is_file())
        self.assertEqual(MemoryEnv.instances, [])

    def test_ranking_branch(self):
        self.config['evaluator']['tasks'].append('policy_ranking')
        self.config['evaluator']['condition_matrix']['policy_ranking'] = copy.deepcopy(
            self.config['evaluator']['condition_matrix']['closed_loop'])
        path = run_benchmark(self.config, ROOT, MemoryEnv)
        metrics = json.loads((path/'metrics.json').read_text())['metrics']
        self.assertEqual(len(metrics['policy_ranking']), 2)
        self.assertTrue(all(r['requested_policies'] == 3 for r in metrics['policy_ranking']))

    def test_config_driven_robustness(self):
        full = load_config(ROOT/'configs/benchmark/boptest_week6.yaml', ROOT)
        evaluator = self.config['evaluator']
        evaluator['tasks'] = ['robustness']
        evaluator['fault_conditions'] = ['noise_0p1', 'missing_1', 'goal_plus1']
        evaluator['faults'] = {k: full['evaluator']['faults'][k] for k in evaluator['fault_conditions']}
        evaluator['condition_matrix'] = {'robustness': dict(episodes=['id_test_day160'], conditions=evaluator['fault_conditions'])}
        evaluator['baseline'] = 'none'
        path = run_benchmark(self.config, ROOT, MemoryEnv)
        metrics = json.loads((path/'metrics.json').read_text())['metrics']
        self.assertEqual(len(metrics['closed_loop']), 6)

    def test_new_mpc_matches_week6_default_search(self):
        from controllers.boptest_mpc import BoptestShootingMPC
        from benchmark.loaders import load_models
        cfg = self.config
        history = np.ones((16, 12))
        history[:, :3] = [21., 35., 30.]
        model = load_models(cfg, ROOT)['persistence']
        h = cfg['horizon']
        legacy_cfg = dict(action=dict(experiment_bounds_c=[18., 24.]), control=dict(
            horizon=h['mpc_steps'], candidates=h['candidates'], action_hold_steps=h['action_hold_steps'],
            time_budget_s=5., comfort_weight=1., energy_weight=1., action_change_weight=.02))
        args = (history, 86400, np.full(h['mpc_steps'], 20.5), np.full(h['mpc_steps'], 24.), 96, 21.)
        actual, detail = ConfiguredMPC(model, cfg).act(*args)
        expected, old_detail = BoptestShootingMPC(model, legacy_cfg).act(*args)
        self.assertEqual(actual, expected)
        self.assertEqual(detail['fallback_reason'], old_detail['fallback_reason'])
        self.assertAlmostEqual(detail['predicted_plan_cost'], old_detail['predicted_plan_cost'])


if __name__ == '__main__':
    unittest.main()
