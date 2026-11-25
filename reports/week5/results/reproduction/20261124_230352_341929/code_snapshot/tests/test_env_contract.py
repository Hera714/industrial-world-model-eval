# 总说明：检验环境接口、随机种子复现、动作响应和回合边界。
import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from simulators.simple_thermal import SimpleThermalEnv, ThermalConfig

class ThermalTest(unittest.TestCase):
    # 使用 Gymnasium 官方检查器验证接口约定。
    def test_gym_contract(self):
        check_env(SimpleThermalEnv(),skip_render_check=True)
    # 相同种子轨迹相同；改变测量噪声不应改变真实状态。
    def test_repeatable_and_noise_streams(self):
        a=SimpleThermalEnv(); b=SimpleThermalEnv()
        oa,ia=a.reset(seed=42); ob,ib=b.reset(seed=42)
        np.testing.assert_array_equal(oa,ob)
        for _ in range(10):
            x=a.step(np.array([2.0],dtype=np.float32)); y=b.step(np.array([2.0],dtype=np.float32))
            np.testing.assert_array_equal(x[0],y[0])
            for k in ('true_state_c','process_noise_c','measurement_noise_c'): np.testing.assert_array_equal(x[4][k],y[4][k])
        c=SimpleThermalEnv(ThermalConfig(measurement_noise_std=0))
        c.reset(seed=42); d=SimpleThermalEnv(); d.reset(seed=42)
        for _ in range(10):
            ci=c.step([2])[4]; di=d.step([2])[4]
            np.testing.assert_array_equal(ci['true_state_c'],di['true_state_c'])
            self.assertEqual(ci['outdoor_used_c'],di['outdoor_used_c'])
    # 相同状态和扰动下，较大加热功率应提高下一步室温。
    def test_heating_effect(self):
        a=SimpleThermalEnv(); b=SimpleThermalEnv(); a.reset(seed=1); b.reset(seed=1)
        low=a.step([0])[4]; high=b.step([5])[4]
        self.assertGreater(high['true_state_c'][0],low['true_state_c'][0])
        self.assertEqual(low['outdoor_used_c'],high['outdoor_used_c'])
        np.testing.assert_array_equal(low['process_noise_c'],high['process_noise_c'])
    # 非法动作不得推进环境；时间上限应返回截断而非自然终止。
    def test_invalid_actions_and_end(self):
        env=SimpleThermalEnv(ThermalConfig(episode_steps=2))
        with self.assertRaises(RuntimeError): env.step([1])
        env.reset(seed=2)
        for a in ([-1],[6],[float('nan')],[float('inf')],[],[1,2]):
            with self.assertRaises(ValueError): env.step(a)
        self.assertEqual(env.step_count,0)
        _,_,term,trunc,_=env.step([1]); self.assertFalse(term or trunc)
        _,_,term,trunc,_=env.step([1]); self.assertFalse(term); self.assertTrue(trunc)
        with self.assertRaises(RuntimeError): env.step([1])
if __name__=='__main__': unittest.main()
