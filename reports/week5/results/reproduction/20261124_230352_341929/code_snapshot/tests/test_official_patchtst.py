import sys, unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/PatchTST/PatchTST_self_supervised'))
from src.callback.patch_mask import create_patch, random_masking
from src.models.patchTST import PatchTST

class OfficialPatchTest(unittest.TestCase):
    def test_nonoverlap_and_mask(self):
        x=torch.arange(144*5,dtype=torch.float32).reshape(1,144,5)
        p,n=create_patch(x,16,16); self.assertEqual(tuple(p.shape),(1,9,5,16)); self.assertEqual(n,9)
        torch.manual_seed(42); masked,_,mask,_=random_masking(p,.4)
        self.assertTrue(torch.all(masked[mask.bool()]==0)); self.assertTrue(torch.equal(masked[~mask.bool()],p[~mask.bool()]))
        self.assertTrue(torch.equal(p[0,0,0],x[0,:16,0])); self.assertTrue(torch.equal(p[0,1,0],x[0,16:32,0]))
    def test_channel_independence(self):
        torch.manual_seed(1); model=PatchTST(c_in=5,target_dim=24,patch_len=16,stride=16,num_patch=9,n_layers=1,n_heads=4,d_model=32,d_ff=64,shared_embedding=True,head_type='prediction',res_attention=False).eval()
        x=torch.randn(2,9,5,16); changed=x.clone(); changed[:,:,1:,:]+=10
        with torch.no_grad(): a=model(x)[:,:,0]; b=model(changed)[:,:,0]
        torch.testing.assert_close(a,b)
if __name__=='__main__': unittest.main()
