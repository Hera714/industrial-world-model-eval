"""官方 PatchTST 的预测诊断和基于验证残差的经验区间。"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_week1_intervals import residual_interval

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'external/PatchTST/PatchTST_self_supervised'))
from src.models.patchTST import PatchTST
from src.callback.patch_mask import create_patch

def load(name):
    with np.load(ROOT/f'data/processed/week1/{name}.npz',allow_pickle=False) as f: return {k:f[k] for k in f.files}

def main():
    out=ROOT/'reports/week1_p1/official_patchtst'; cfg=json.loads((out/'config.json').read_text()); meta=json.loads((ROOT/'data/processed/week1/metadata.json').read_text())
    val,test=load('val'),load('test')
    with np.load(out/'test_predictions.npz',allow_pickle=False) as f: saved={k:f[k] for k in f.files}
    if not np.array_equal(test['target_start'],saved['target_start']): raise ValueError('测试窗口时间不匹配')
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=PatchTST(**cfg['model'],head_type='prediction').to(device)
    model.load_state_dict(torch.load(out/'finetuned.pt',map_location=device,weights_only=True)); model.eval()
    def predict(data):
        values=[]
        with torch.no_grad():
            for i in range(0,len(data['X']),256):
                x=torch.from_numpy(data['X'][i:i+256]).to(device); patches,_=create_patch(x,cfg['patch_length'],cfg['stride']); values.append(model(patches)[:,:,0].cpu().numpy())
        return np.concatenate(values)*meta['scaler']['scale']['T1']+meta['scaler']['mean']['T1']
    pv=predict(val); pt=predict(test)
    if not np.allclose(pt,saved['predicted'],atol=2e-4,rtol=1e-5): raise ValueError('模型预测与保存结果不一致')
    actual_val=val['y']*meta['scaler']['scale']['T1']+meta['scaler']['mean']['T1']; actual=saved['actual']; point=saved['predicted']
    low,high,qlo,qhi=residual_interval(actual_val,pv,point); coverage=((actual>=low)&(actual<=high)).mean(axis=0); width=(high-low).mean(axis=0); h=np.arange(1,actual.shape[1]+1)
    table=pd.DataFrame({'horizon':h,'minutes':10*h,'coverage':coverage,'mean_width_c':width,'residual_q05_c':qlo,'residual_q95_c':qhi})
    table.to_csv(out/'interval_metrics.csv',index=False)
    np.savez_compressed(out/'interval_predictions.npz',actual=actual,predicted=point,lower=low,upper=high,target_start=saved['target_start'])
    fig,ax=plt.subplots(figsize=(9,4)); ax.plot(h,actual[0],label='actual',color='black'); ax.plot(h,point[0],label='official PatchTST'); ax.fill_between(h,low[0],high[0],alpha=.25,label='nominal 90%'); ax.set(xlabel='Horizon (10-min steps)',ylabel='T1 (°C)',title='Official PatchTST: test window 0'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'interval_example.png',dpi=160); plt.close(fig)
    error=point-actual; mae=np.abs(error).mean(axis=0); rmse=np.sqrt((error**2).mean(axis=0))
    fig,ax=plt.subplots(2,2,figsize=(11,8)); ax[0,0].plot(h,mae,label='MAE'); ax[0,0].plot(h,rmse,label='RMSE'); ax[0,0].set(xlabel='Horizon (10-min steps)',ylabel='Error (°C)',title='Prediction error'); ax[0,0].legend()
    ax[0,1].scatter(actual[:,0],point[:,0],s=3,alpha=.15); limits=[min(actual[:,0].min(),point[:,0].min()),max(actual[:,0].max(),point[:,0].max())]; ax[0,1].plot(limits,limits,'k--'); ax[0,1].set(xlabel='Actual T1 (°C)',ylabel='Predicted T1 (°C)',title='One-step actual vs predicted')
    ax[1,0].plot(h,coverage*100); ax[1,0].axhline(90,color='black',linestyle='--'); ax[1,0].set(xlabel='Horizon (10-min steps)',ylabel='Coverage (%)',title='Interval coverage')
    ax[1,1].plot(h,width); ax[1,1].set(xlabel='Horizon (10-min steps)',ylabel='Width (°C)',title='Average interval width')
    for a in ax.flat: a.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'prediction_diagnostics.png',dpi=160); plt.close(fig)
    print(table[table.horizon.isin([1,6,12,24])].to_string(index=False))

if __name__=='__main__': main()
