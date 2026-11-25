"""总说明：配置在网络启动前拒绝缺字段、单位/范围/哈希错误及数据泄漏。"""
from pathlib import Path
import copy,tempfile,unittest
from unittest.mock import patch
import yaml
from benchmark.config import load_config,validate_config,ConfigurationError

ROOT=Path(__file__).resolve().parents[1]


class BenchmarkConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke=yaml.safe_load((ROOT/'configs/benchmark/boptest_smoke.yaml').read_text())

    def test_valid_configs_without_network(self):
        with patch('socket.socket.connect',side_effect=AssertionError('Unexpected network')) as connection:
            for name,count in [('boptest_smoke.yaml',2),('boptest_week6.yaml',11)]:
                config=load_config(ROOT/'configs/benchmark'/name,ROOT)
                self.assertEqual(len(config['model']['instances']),count)
            connection.assert_not_called()

    def test_required_top_level_fields(self):
        for field in self.smoke:
            with self.subTest(field=field):
                config=copy.deepcopy(self.smoke);del config[field]
                with self.assertRaisesRegex(ConfigurationError,'missing required field'):
                    validate_config(config,ROOT,verify_files=False)

    def test_nested_field_error_identifies_path(self):
        config=copy.deepcopy(self.smoke);del config['env']['action']['api_unit']
        with self.assertRaisesRegex(ConfigurationError,'env.action.api_unit'):
            validate_config(config,ROOT,False)

    def test_unit_errors(self):
        for section,field,value in [('env','step_seconds',600),('metrics','units',{'energy':'W'}),('model','observation_units',['K']*5)]:
            with self.subTest(field=field):
                config=copy.deepcopy(self.smoke);config[section][field]=value
                with self.assertRaises(ConfigurationError):validate_config(config,ROOT,False)

    def test_range_and_unknown_errors(self):
        mutations=[('horizon','prediction_steps',[1,9]),('horizon','candidates',True),
            ('horizon','prediction_steps',[4,1]),('horizon','mpc_steps',float('nan')),
            ('env','episode_steps',0),('model','input_features',13)]
        for section,field,value in mutations:
            with self.subTest(field=field):
                config=copy.deepcopy(self.smoke);config[section][field]=value
                with self.assertRaises(ConfigurationError):validate_config(config,ROOT,False)
        config=copy.deepcopy(self.smoke);config['horizon']['canddiates']=32
        with self.assertRaisesRegex(ConfigurationError,'unknown fields'):validate_config(config,ROOT,False)

    def test_hash_mismatch_is_detected(self):
        config=copy.deepcopy(self.smoke);config['model']['instances'][0]['checkpoint']['sha256']='0'*64
        with self.assertRaisesRegex(ConfigurationError,'digest mismatch'):validate_config(config,ROOT)

    def test_checkpoint_kind_is_verified(self):
        config=copy.deepcopy(self.smoke);config['model']['instances'][0]['kind']='arx'
        with self.assertRaisesRegex(ConfigurationError,'kind/history mismatch'):validate_config(config,ROOT)

    def test_training_episode_cannot_be_evaluation(self):
        config=copy.deepcopy(self.smoke);config['dataset']['evaluation_episodes']=['train_day040']
        for task in config['evaluator']['condition_matrix'].values():task['episodes']=['train_day040']
        with self.assertRaisesRegex(ConfigurationError,'leakage'):validate_config(config,ROOT)

    def test_advance_retry_forbidden(self):
        config=copy.deepcopy(self.smoke);config['failure_policy']['retries']['advance_requests']=1
        with self.assertRaisesRegex(ConfigurationError,'advancing cannot be retried'):validate_config(config,ROOT,False)

    def test_output_path_traversal(self):
        config=copy.deepcopy(self.smoke);config['output']['directory']='../other_project'
        with self.assertRaisesRegex(ConfigurationError,'stay inside project'):validate_config(config,ROOT,False)

    def test_duplicate_yaml_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.yaml';path.write_text('schema_version: "1.0"\nschema_version: "2.0"\n')
            with self.assertRaisesRegex(ConfigurationError,'duplicate key'):load_config(path,ROOT,False)


if __name__=='__main__':unittest.main()
