# 总说明：检验配对回合的外部扰动与噪声相同，并确保两个控制器完成回合。
import unittest
import numpy as np
from evaluators.closed_loop import run_episode
from simulators.simple_thermal import ThermalConfig
class PairedEpisodeTest(unittest.TestCase):
    def test_paired_exogenous_streams(self):
        scenario={'seed':7,'initial_air_c':18.,'initial_wall_c':19.}
        c=ThermalConfig(episode_steps=12)
        a,ma,_=run_episode('rule',scenario,c); b,mb,_=run_episode('mpc',scenario,c)
        for key in ('outdoor_used_c','outdoor_next_c','process_air_c','process_wall_c','measurement_air_c','measurement_wall_c','measurement_outdoor_c'):
            np.testing.assert_array_equal([x[key] for x in a],[x[key] for x in b])
        self.assertEqual(ma['steps'],12); self.assertEqual(mb['steps'],12)
        self.assertTrue(a[-1]['truncated'] and b[-1]['truncated'])
if __name__=='__main__': unittest.main()
