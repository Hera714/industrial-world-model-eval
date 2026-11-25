"""使用保存的 PatchTST 权重和验证残差评估经验预测区间。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_week1_intervals import residual_interval

class SavedPatchTST(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.patch=cfg['patch_length']; self.stride=cfg['stride']; d=cfg['d_model']
        n=(cfg['history']-self.patch)//self.stride+1
        self.embed=nn.Linear(self.patch,d); self.pos=nn.Parameter(torch.zeros(1,n,d))
        layer=nn.TransformerEncoderLayer(d,cfg['heads'],dim_feedforward=128,dropout=.1,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,cfg['layers'],enable_nested_tensor=False); self.head=nn.Linear(n*d,cfg['horizon'])
    def forward(self,x):
        patches=x[:,:,0].unfold(1,self.patch,self.stride)
        return self.head(self.encoder(self.embed(patches)+self.pos).flatten(1))

def main():
    root=Path('reports/week1_p1'); out=root/'patchtst_interval'; out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/'patchtst_config.json').read_text()); meta=json.loads(Path('data/processed/week1/metadata.json').read_text()); idx=meta['features'].index('T1')
    def load(name):
        with np.load(f'data/processed/week1/{name}.npz',allow_pickle=False) as f: return {k:f[k] for k in f.files}
    val,test=load('val'),load('test')
    with np.load(root/'patchtst_predictions.npz',allow_pickle=False) as f: saved={k:f[k] for k in f.files}
    if not np.array_equal(test['target_start'],saved['target_start']): raise ValueError('测试窗口不一致')
    model=SavedPatchTST(cfg); model.load_state_dict(torch.load(root/'patchtst.pt',map_location='cpu',weights_only=True)); model.eval()
    def predict(data):
        x=data['X'][:,:,idx:idx+1]; chunks=[]
        with torch.no_grad():
            for start in range(0,len(x),256): chunks.append(model(torch.from_numpy(x[start:start+256])).numpy())
        return np.concatenate(chunks)*meta['scaler']['scale']['T1']+meta['scaler']['mean']['T1']
    vp=predict(val); tp=predict(test)
    if not np.allclose(tp,saved['patchtst_t1'],atol=2e-4,rtol=1e-5): raise ValueError('权重与保存的 PatchTST 预测不一致')
    actual_val=val['y']*meta['scaler']['scale']['T1']+meta['scaler']['mean']['T1']; actual=saved['actual']; tp=saved['patchtst_t1']
    lo,hi,qlo,qhi=residual_interval(actual_val,vp,tp)
    coverage=((actual>=lo)&(actual<=hi)).mean(axis=0); width=(hi-lo).mean(axis=0); h=np.arange(1,actual.shape[1]+1)
    table=pd.DataFrame({'horizon':h,'minutes':h*10,'coverage':coverage,'mean_width_c':width,'residual_q05_c':qlo,'residual_q95_c':qhi})
    table.to_csv(out/'metrics_by_horizon.csv',index=False); np.savez_compressed(out/'predictions.npz',actual=actual,predicted=tp,lower=lo,upper=hi,target_start=test['target_start'])
    fig,ax=plt.subplots(figsize=(9,4)); ax.plot(h,actual[0],label='actual'); ax.plot(h,tp[0],label='PatchTST forecast'); ax.fill_between(h,lo[0],hi[0],alpha=.25,label='nominal 90%'); ax.set(xlabel='Horizon (10-min steps)',ylabel='T1 (°C)',title='PatchTST: test window 0'); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out/'interval_example.png',dpi=160); plt.close(fig)
    print(table[table.horizon.isin([1,6,12,24])].to_string(index=False))
if __name__=='__main__': main()
