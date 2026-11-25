"""检验统一转移的溯源、时间对齐及 CSV 往返。"""
import io
import json
import unittest
import numpy as np
import pandas as pd
from data.collect_trajectories import collect_episodes
from simulators.simple_thermal import ThermalConfig


class DynamicsSchemaTest(unittest.TestCase):
    def test_transition_and_reproducibility(self):
        # 小回合同时覆盖随机与规则两类采集策略。
        config = ThermalConfig(episode_steps=3)
        frame = collect_episodes(2, 42, config)
        pd.testing.assert_frame_equal(frame, collect_episodes(2, 42, config))
        restored = pd.read_csv(io.StringIO(frame.to_csv(index=False)), keep_default_na=False)
        self.assertFalse(restored.duplicated(['episode_id', 'source_step']).any())
        for _, episode in restored.groupby('episode_id'):
            self.assertEqual(len(episode), 3)
            self.assertTrue(episode.iloc[-1].truncated)
            self.assertEqual(episode.iloc[-1].end_reason, 'time_limit')
            for index, (_, row) in enumerate(episode.iterrows()):
                np.testing.assert_allclose(json.loads(row.state), [row.air, row.wall])
                np.testing.assert_allclose(json.loads(row.action_applied), [row.action])
                np.testing.assert_allclose(json.loads(row.exogenous), [row.outdoor, config.internal_load_kw])
                self.assertAlmostEqual(row.next_time-row.time, row['dt'])
                if index:
                    np.testing.assert_allclose(json.loads(episode.iloc[index-1].next_state), json.loads(row.state))
