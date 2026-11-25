"""动作方向/滞后及分组统计测试：检查因果步序、零响应和目标转移功率标签。"""
import unittest
import json
import numpy as np
import pandas as pd
from metrics.action_response import response_metrics
from scripts.decompose_week5 import endpoint_metadata,group_errors
from scripts.action_week5 import simulate_plan
from simulators.simple_thermal import ThermalConfig


class DiagnosticTest(unittest.TestCase):
    def test_direction_amplitude_and_lag(self):
        truth=np.array([0.,1.,2.,4.]); model=truth.copy()
        result=response_metrics(truth,model,0)
        self.assertEqual(result['direction_accuracy'],1.)
        self.assertEqual(result['peak_magnitude_error_c'],0.)
        self.assertEqual(result['true_lag_steps'],2)
        self.assertEqual(result['lag_error_minutes'],0.)
        lagged=np.array([0.,0.,1.,2.])
        self.assertEqual(response_metrics(truth,lagged,0)['lag_error_minutes'],10.)
        self.assertEqual(response_metrics(-truth,-model,0)['lag_error_minutes'],0.)

    def test_no_action_response_is_not_zero_lag_error(self):
        result=response_metrics(np.array([0.,1.,2.,4.]),np.zeros(4),0)
        self.assertEqual(result['direction_accuracy'],0.)
        self.assertEqual(result['lag_status'],'not_reached_within_horizon')
        self.assertTrue(np.isnan(result['lag_error_minutes']))
        self.assertEqual(result['peak_magnitude_error_c'],-4.)

    def test_unidentifiable_and_missing_response(self):
        result=response_metrics(np.zeros(4),np.zeros(4),0)
        self.assertEqual(result['lag_status'],'true_response_too_small')
        result=response_metrics(np.arange(4.),np.array([0.,1.,np.nan,np.nan]),0)
        self.assertEqual(result['valid_steps'],1)
        self.assertEqual(result['lag_status'],'prediction_incomplete')

    def test_real_environment_causal_heating_and_shared_weather(self):
        config=ThermalConfig(episode_steps=3,process_noise_std=0,measurement_noise_std=0,weather_noise_std=0)
        cold,d0=simulate_plan(config,np.zeros(3)); hot,d1=simulate_plan(config,np.full(3,5.))
        np.testing.assert_equal(d0,d1)
        self.assertGreater(hot[1,0],cold[1,0])
        self.assertEqual(hot[1,1],cold[1,1])
        self.assertGreater(hot[2,1],cold[2,1])

    def test_metadata_uses_target_transition_power_and_episode_initial(self):
        frame=pd.DataFrame(dict(episode_id=['e']*5,source_step=range(5),
            state=[json.dumps([18.+i,19.]) for i in range(5)],policy=['random']*5,
            action_applied=[json.dumps([i]) for i in range(5)]))
        origins=pd.DataFrame(dict(episode_id=['e','e'],source_step=[0,1]))
        result=endpoint_metadata(frame,origins,4)
        np.testing.assert_equal(result.target_power_kw.to_numpy(),[3.,4.])
        self.assertEqual(result.iloc[1].target_power_bin,'>=4')
        self.assertTrue(result.initial_air_bin.eq('<19').all())

    def test_group_statistics_keep_episode_weighting(self):
        rows=[]
        for ep,errors in [('a',[1.,1.,1.]),('b',[3.])]:
            for i,error in enumerate(errors):
                rows.append(dict(model='linear',family='linear',seed=-1,mode='free_running',condition='recorded',
                    horizon=1,variable='air',episode_id=ep,origin_index=i,target_power_bin='<1',
                    policy='rule',initial_air_bin='<19',valid_prediction=True,complete_prefix=True,
                    failed_by_h=False,abs_error_c=error,squared_error_c=error**2))
        episodes,summary,seeds=group_errors(pd.DataFrame(rows))
        row=summary[summary.grouping.eq('all')].iloc[0]
        self.assertEqual(row.rmse_c_mean,2.)
        self.assertEqual(row.requested,4)


if __name__=='__main__': unittest.main()
