import unittest
import numpy as np
from run_week1_intervals import residual_interval
class IntervalTest(unittest.TestCase):
    def test_per_horizon_quantiles(self):
        y=np.array([[1,10],[2,20],[3,30]],float); p=np.zeros_like(y); t=np.array([[10,100]],float)
        lo,hi,qlo,qhi=residual_interval(y,p,t,alpha=.5)
        np.testing.assert_allclose(qlo,[1.5,15]); np.testing.assert_allclose(qhi,[2.5,25]); np.testing.assert_allclose(lo,[[11.5,115]]); np.testing.assert_allclose(hi,[[12.5,125]])
    def test_shape_rejected(self):
        with self.assertRaises(ValueError): residual_interval(np.ones((3,2)),np.ones((3,1)),np.ones((1,2)))
if __name__=='__main__': unittest.main()
