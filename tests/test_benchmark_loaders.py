"""总说明：统一加载的真实检查点回归测试，检查维度、哈希、设备与窗口独立性。"""
from pathlib import Path
import copy
import hashlib
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from benchmark.config import ConfigurationError, load_config, validate_config
from benchmark.loaders import load_models

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkLoadersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / 'configs/benchmark/boptest_week6.yaml', ROOT)
        cls.models = load_models(cls.config, ROOT)

    def test_all_models_shapes_and_legacy_equivalence(self):
        self.assertEqual(len(self.models), 11)
        self.assertEqual({m.kind for m in self.models.values()},
                         {'persistence', 'arx', 'mlp', 'residual_mlp', 'gru'})
        for name, model in self.models.items():
            with self.subTest(name=name):
                history = np.tile(model.model.mean, (16, 1))
                actual = model.predict(history)
                self.assertEqual(actual.shape, (6,))
                np.testing.assert_allclose(actual, model.model.predict(history), rtol=1e-6, atol=1e-7)
                batch = model.predict(np.stack([history, history]))
                self.assertEqual(batch.shape, (2, 6))
                np.testing.assert_allclose(batch[0], actual, rtol=1e-5, atol=1e-5)
                transition = model.predict_transition(history)
                self.assertEqual(transition['next_observation'].shape, (5,))

    def test_gru_windows_do_not_share_hidden_state(self):
        for model in self.models.values():
            if model.kind != 'gru':
                continue
            history = np.tile(model.model.mean, (16, 1))
            first = model.predict(history)
            model.predict(history + .5)
            model.reset()
            np.testing.assert_array_equal(first, model.predict(history))

    def test_invalid_histories(self):
        for model in self.models.values():
            for history in [np.zeros((15, 12)), np.zeros((16, 13)),
                            np.zeros((0, 16, 12)), np.full((16, 12), np.nan)]:
                with self.assertRaises(ValueError):
                    model.predict(history)

    def test_bad_hash_before_loading_or_network(self):
        config = copy.deepcopy(self.config)
        config['model']['instances'][0]['checkpoint']['sha256'] = '0' * 64
        with patch('socket.socket.connect') as connection, patch('benchmark.loaders.load_boptest_dynamics') as load:
            with self.assertRaisesRegex(ConfigurationError, 'digest mismatch'):
                load_models(config, ROOT)
            load.assert_not_called()
            connection.assert_not_called()

    def test_checkpoint_dimensions_and_scale(self):
        # 新文件重新计算正确哈希，确认拒绝的是内容维度而非哈希。
        spec = next(s for s in self.config['model']['instances'] if s['kind'] == 'arx')
        data = torch.load(ROOT / spec['checkpoint']['path'], weights_only=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory) / 'bad.pt'
            for field, value in [('mean', [0.] * 11), ('scale', [0.] * 12), ('coef', [[0.] * 6] * 12)]:
                corrupted = copy.deepcopy(data)
                corrupted[field] = value
                torch.save(corrupted, target)
                config = copy.deepcopy(self.config)
                selected = next(s for s in config['model']['instances'] if s['kind'] == 'arx')
                selected['checkpoint'] = {'path': str(target.relative_to(ROOT)),
                                          'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
                with self.assertRaisesRegex(ConfigurationError, field):
                    validate_config(config, ROOT)

    def test_unavailable_cuda_rejected(self):
        config = copy.deepcopy(self.config)
        config['model']['device'] = 'cuda:0'
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ConfigurationError, 'unavailable'):
                load_models(config, ROOT)


if __name__ == '__main__':
    unittest.main()
