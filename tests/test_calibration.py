"""概率校准单元测试：阶统计量、变量/步长隔离、失败分母及禁止测试集拟合。"""
import unittest
import numpy as np
from metrics.calibration import ensemble_moments,fit_interval_scale,interval_arrays,interval_metrics


class CalibrationTest(unittest.TestCase):
    def members(self,n=9,h=2):
        return np.broadcast_to(np.array([-1.,0.,1.])[:,None,None,None],(3,n,h,2)).copy()

    def test_exact_order_statistic_per_horizon_variable(self):
        members=self.members()
        target=np.broadcast_to(np.arange(1.,10.)[:,None,None],(9,2,2)).copy()
        target[:,1,0]*=2; target[:,0,1]*=3
        result=fit_interval_scale(members,target,nominal=.9)
        np.testing.assert_equal(result['rank'],9)
        np.testing.assert_equal(result['q'],[[9,27],[18,9]])

    def test_zero_spread_floor_and_frozen_application(self):
        members=np.zeros((3,9,1,2)); target=np.full((9,1,2),.02)
        fit=fit_interval_scale(members,target,scale_floor_c=.01)
        np.testing.assert_allclose(fit['q'],2.)
        interval=interval_arrays(np.zeros((3,4,1,2)),fit['q'],.01)
        np.testing.assert_allclose(interval['lower'],-.02)
        np.testing.assert_allclose(interval['upper'],.02)

    def test_test_fitting_rejected_and_insufficient_scores(self):
        with self.assertRaises(ValueError): fit_interval_scale(self.members(),np.zeros((9,2,2)),split='test')
        result=fit_interval_scale(self.members(n=1),np.zeros((1,2,2)))
        self.assertTrue(np.isnan(result['q']).all())
        self.assertTrue((result['status']=='insufficient_valid_scores').all())

    def test_failed_member_invalidates_interval_without_dropping_denominator(self):
        members=self.members(n=2,h=1); members[0,1,0,0]=np.nan
        mean,sd,scale=ensemble_moments(members,.01)
        self.assertTrue(np.isnan(mean[1,0,0]))
        result=interval_metrics(np.array([0.,0.]),np.array([-1.,np.nan]),np.array([1.,np.nan]))
        self.assertEqual(result['requested'],2)
        self.assertEqual(result['invalid'],1)
        self.assertEqual(result['coverage_all'],.5)
        self.assertEqual(result['coverage_valid'],1.)
        self.assertEqual(result['mean_width_c'],2.)

    def test_boundaries_inclusive(self):
        result=interval_metrics(np.array([-1.,1.,2.]),np.array([-1.,-1.,-1.]),np.array([1.,1.,1.]))
        self.assertEqual(result['covered'],2)


if __name__=='__main__': unittest.main()
