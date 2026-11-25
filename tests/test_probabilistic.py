"""经验 CRPS 核验：手算、精确CDF积分、重复成员、单位变换和失败分母语义。"""
import unittest
import numpy as np
from metrics.probabilistic import empirical_crps,empirical_crps_components


class EmpiricalCRPSTest(unittest.TestCase):
    def test_three_member_hand_calculation_not_fair_estimator(self):
        result=empirical_crps_components(np.array([0.]),np.array([[-1.],[0.],[1.]]))
        self.assertAlmostEqual(result['mean_member_abs_error_c'][0],2/3)
        self.assertAlmostEqual(result['pair_half_dispersion_c'][0],4/9)
        self.assertAlmostEqual(result['empirical_crps_c'][0],2/9)

    def test_point_mass_equals_absolute_error(self):
        actual=np.array([1.,1.]); prediction=np.array([[0.,3.]])
        np.testing.assert_equal(empirical_crps(actual,prediction),[1.,2.])
        np.testing.assert_equal(empirical_crps(actual,np.repeat(prediction,3,axis=0)),[1.,2.])

    def test_empirical_cdf_unchanged_by_member_replication(self):
        members=np.array([[-1.,2.],[0.,3.],[1.,5.]])
        target=np.array([.2,4.])
        np.testing.assert_allclose(empirical_crps(target,members),empirical_crps(target,np.repeat(members,2,axis=0)))

    def test_exact_piecewise_cdf_integral(self):
        members=np.array([-2.,.5,4.]); target=1.2
        knots=np.unique(np.r_[members,target]); mids=(knots[:-1]+knots[1:])/2
        cdf=(members[:,None]<=mids).mean(axis=0); observed=(target<=mids).astype(float)
        integral=np.sum(np.diff(knots)*(cdf-observed)**2)
        self.assertAlmostEqual(float(empirical_crps(np.array(target),members)),integral)

    def test_no_partial_member_renormalization(self):
        result=empirical_crps(np.zeros(2),np.array([[0.,0.],[1.,np.nan],[2.,3.]]))
        self.assertTrue(np.isfinite(result[0]))
        self.assertTrue(np.isnan(result[1]))

    def test_units_translation_and_scaling(self):
        target=np.array([.2,4.]); members=np.array([[-1.,2.],[0.,3.],[1.,5.]])
        score=empirical_crps(target,members)
        np.testing.assert_allclose(empirical_crps(target+273.15,members+273.15),score)
        np.testing.assert_allclose(empirical_crps(2*target,2*members),2*score)

    def test_invalid_inputs(self):
        for target,members in [(np.array([]),np.empty((3,0))),
                               (np.zeros(2),np.zeros((3,3))),
                               (np.array([np.inf]),np.zeros((3,1)))]:
            with self.assertRaises(ValueError): empirical_crps(target,members)


if __name__=='__main__': unittest.main()
