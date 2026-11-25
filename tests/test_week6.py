"""总说明：BOPTEST 世界模型历史隔离、动作时序、MPC 后备和独立成本的回归测试。"""
import unittest,copy
import numpy as np
import pandas as pd
import torch
from models.boptest_dynamics import BoptestHistoryModel,exogenous_features
from controllers.boptest_mpc import BoptestShootingMPC,stage_cost
from scripts.evaluate_week6_open import rollout


class Dummy:
    length=4
    def __init__(self,negative=False):self.negative=negative
    def predict(self,history):
        value=np.asarray(history);single=value.ndim==2
        if single:value=value[None]
        result=np.column_stack([value[:,-1,:5],np.ones(len(value))*(-1 if self.negative else .2)])
        return result[0] if single else result


class Week6Test(unittest.TestCase):
    def test_gru_independent_windows(self):
        torch.set_num_threads(1);model=BoptestHistoryModel('gru',4)
        x=np.ones((2,4,12));a=model.predict(x);model.predict(x*2);np.testing.assert_array_equal(a,model.predict(x))
        np.testing.assert_allclose(a[0],model.predict(x[0]),rtol=1e-5,atol=1e-7)
    def test_input_validation(self):
        model=BoptestHistoryModel('persistence',4)
        for values in (np.ones((3,12)),np.full((4,12),np.nan)):
            with self.assertRaises(ValueError):model.predict(values)
    def test_energy_is_not_setpoint(self):
        model=BoptestHistoryModel('persistence',4);x=np.ones((4,12));x[-1,4]=2.;x[-1,5]=24.
        self.assertEqual(model.predict(x)[-1],.5)
    def test_no_future_measurements_in_free_rollout(self):
        features=np.ones((15,12));features[:,:3]=21.;features[:,4]=2.
        altered=features.copy();altered[5:,:5]=999.
        a=rollout(Dummy(),features,4,5,'persistence_weather')[0]
        b=rollout(Dummy(),altered,4,5,'persistence_weather')[0]
        np.testing.assert_array_equal(a,b)
    def test_rejected_candidates_release(self):
        config=dict(action=dict(experiment_bounds_c=[18,24]),control=dict(horizon=2,candidates=8,action_hold_steps=1,time_budget_s=5.,comfort_weight=1,energy_weight=1,action_change_weight=.02))
        controller=BoptestShootingMPC(Dummy(True),config)
        history=np.ones((4,12));history[:,:3]=21.
        action,info=controller.act(history,0,np.full(2,20),np.full(2,24),0,21)
        self.assertIsNone(action);self.assertEqual(info['fallback_reason'],'no_valid_candidates')
        config['control']['time_budget_s']=1e-12
        action,info=BoptestShootingMPC(Dummy(),config).act(history,0,np.full(2,20),np.full(2,24),0,21)
        self.assertIsNone(action);self.assertEqual(info['fallback_reason'],'timeout')
    def test_cost_independent_comfort_standard(self):
        self.assertEqual(stage_cost(22,1,21,24,20,20),1)
        self.assertGreater(stage_cost(19,1,21,24,20,20),1)
    def test_known_calendar_features(self):
        a=exogenous_features(np.array([10,0]),np.asarray(0.))
        b=exogenous_features(np.array([10,0]),np.asarray(86400.))
        np.testing.assert_allclose(a[:4],b[:4],atol=1e-12)


if __name__=='__main__':unittest.main()
