# 总说明：检查线性辨识能恢复已知动力学，且候选预测支持批处理。
import unittest
import numpy as np
from models.system_identification import LinearDynamics
class IdentificationTest(unittest.TestCase):
    def test_recover_linear_system(self):
        rng=np.random.default_rng(9); x=rng.normal(size=(100,2)); u=rng.normal(size=100); d=rng.normal(size=100)
        y=np.column_stack([.9*x[:,0]+.1*x[:,1]+.3*u+.2*d+1, .1*x[:,0]+.8*x[:,1]-.2*d])
        model=LinearDynamics().fit(x,u,d,y)
        np.testing.assert_allclose(model.predict_next(x,u,d),y,atol=1e-12)
    def test_shape(self):
        rng=np.random.default_rng(5); x=rng.normal(size=(20,2)); m=LinearDynamics().fit(x,np.arange(20),np.arange(20),x)
        self.assertEqual(m.predict_next(x[:3],np.ones(3),5).shape,(3,2))
if __name__=='__main__': unittest.main()
