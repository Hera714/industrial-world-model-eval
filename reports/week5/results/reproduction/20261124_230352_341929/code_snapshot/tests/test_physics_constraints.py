"""用明显异常输出检查投影触发和修正后边界。"""
import unittest
import numpy as np
from models.dynamics_base import PersistenceDynamics
from models.physics_constraints import RateConstrainedDynamics


class BrokenDynamics(PersistenceDynamics):
    def _predict(self,state,action,exogenous):
        return state+100


class RateConstraintTest(unittest.TestCase):
    def test_projection(self):
        wrapper=RateConstrainedDynamics(BrokenDynamics())
        state=np.array([20.,19.])
        output,detail=wrapper.predict_with_diagnostics(state,[5.],[6.,.3])
        self.assertTrue(detail['triggered'].all())
        self.assertTrue(np.all(output-state <= detail['high_delta']+1e-12))
        self.assertTrue(np.all(output-state >= detail['low_delta']-1e-12))
