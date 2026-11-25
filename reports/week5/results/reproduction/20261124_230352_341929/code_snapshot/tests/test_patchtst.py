import unittest
import numpy as np
from models.patchtst import PatchTST
class PatchTSTTest(unittest.TestCase):
    def test_shape_and_finite(self):
        r=np.random.default_rng(5); x=r.normal(size=(20,16,1)).astype('float32'); y=r.normal(size=(20,4)).astype('float32')
        m=PatchTST(history=16,horizon=4,patch=4,stride=4,d_model=8,heads=2,layers=1,epochs=1,patience=1,batch=8,device='cpu')
        m.fit(x[:12],y[:12],x[12:16],y[12:16]); p=m.predict(x[16:]); self.assertEqual(p.shape,(4,4)); self.assertTrue(np.isfinite(p).all())
if __name__=='__main__': unittest.main()
