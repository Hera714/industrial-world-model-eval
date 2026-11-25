"""用验证残差为已训练的 Week 1 基线分别构造经验预测区间。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.linear_model import Ridge

DATA=Path('data/processed/week1'); BASE=Path('reports/week1_baselines'); OUT=Path('reports/week1_p1/model_intervals')

def residual_interval(validation_actual, validation_prediction, test_prediction, alpha=.1):
    """每步使用验证集残差的双侧分位数，固定后应用于测试集。"""
    a=np.asarray(validation_actual); p=np.asarray(validation_prediction); t=np.asarray(test_prediction)
    if a.ndim!=2 or p.shape!=a.shape or t.ndim!=2 or t.shape[1]!=a.shape[1] or not all(np.isfinite(x).all() for x in (a,p,t)) or not 0<alpha<1:
        raise ValueError('输入形状、数值或 alpha 不合法')
    qlo,qhi=np.quantile(a-p,[alpha/2,1-alpha/2],axis=0)
    return t+qlo,t+qhi,qlo,qhi

def load(name):
    with np.load(DATA/f'{name}.npz',allow_pickle=False) as f: return {k:f[k] for k in f.files}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((DATA/'metadata.json').read_text()); config=json.loads((BASE/'config.json').read_text())
    train,val,test=(load(n) for n in ('train','val','test'))
    with np.load(BASE/'test_predictions.npz',allow_pickle=False) as f: saved={k:f[k] for k in f.files}
    if not np.array_equal(test['target_start'],saved['target_start']): raise ValueError('测试窗口与已有预测不一致')
    mu=meta['scaler']['mean']['T1']; scale=meta['scaler']['scale']['T1']; i=meta['features'].index('T1'); horizon=meta['horizon']
    to_c=lambda y:y*scale+mu
    actual_val=to_c(val['y']); actual_test=to_c(test['y'])
    if not np.allclose(actual_test,saved['actual'],atol=1e-5): raise ValueError('测试目标不一致')
    vpred={'persistence':np.repeat(to_c(val['X'][:,-1,i])[:,None],horizon,axis=1)}
    linear=Ridge(alpha=config['linear']['selected_alpha']).fit(train['X'].reshape(len(train['X']),-1).astype(np.float64),train['y'])
    vpred['linear']=to_c(linear.predict(val['X'].reshape(len(val['X']),-1)))
    reconstructed=to_c(linear.predict(test['X'].reshape(len(test['X']),-1)))
    if not np.allclose(reconstructed,saved['linear'],atol=1e-5,rtol=1e-5): raise ValueError('线性模型与已保存结果不一致，请重新运行基线')
    for name,cfg in config['mlp'].items():
        model=nn.Sequential(nn.Flatten(),nn.Linear(train['X'].shape[1]*train['X'].shape[2],cfg['hidden']),nn.ReLU(),nn.Linear(cfg['hidden'],cfg['hidden']),nn.ReLU(),nn.Linear(cfg['hidden'],horizon))
        model.load_state_dict(torch.load(BASE/f'{name}.pt',map_location='cpu',weights_only=True)); model.eval()
        with torch.no_grad():
            vpred[name]=to_c(model(torch.from_numpy(val['X'])).numpy())
            reconstructed=to_c(model(torch.from_numpy(test['X'])).numpy())
        if not np.allclose(reconstructed,saved[name],atol=2e-4,rtol=1e-5): raise ValueError(f'{name} 权重与保存的测试预测不一致')
    rows=[]; outputs={'actual':actual_test,'target_start':test['target_start']}; bounds={}
    for name,vp in vpred.items():
        point=saved[name]; lo,hi,qlo,qhi=residual_interval(actual_val,vp,point); bounds[name]=(lo,hi,point)
        covered=((actual_test>=lo)&(actual_test<=hi)).mean(axis=0); width=(hi-lo).mean(axis=0)
        for j in range(horizon): rows.append({'model':name,'horizon':j+1,'minutes':10*(j+1),'coverage':covered[j],'mean_width_c':width[j],'residual_q05_c':qlo[j],'residual_q95_c':qhi[j]})
        outputs[name+'_pred']=point; outputs[name+'_lower']=lo; outputs[name+'_upper']=hi
    df=pd.DataFrame(rows); df.to_csv(OUT/'metrics_by_horizon.csv',index=False); df[df.horizon.isin([1,6,12,24])].to_csv(OUT/'metrics_key_horizons.csv',index=False)
    np.savez_compressed(OUT/'predictions.npz',**outputs)
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True,sharey=True); h=np.arange(1,horizon+1)
    for ax,(name,(lo,hi,point)) in zip(axes.flat,bounds.items()):
        ax.plot(h,actual_test[0],label='actual',color='black'); ax.plot(h,point[0],label='forecast'); ax.fill_between(h,lo[0],hi[0],alpha=.25,label='nominal 90%'); ax.set(title=name,xlabel='Horizon (10-min steps)',ylabel='T1 (°C)'); ax.grid(alpha=.2); ax.legend()
    axes.flat[-1].set_visible(False); fig.tight_layout(); fig.savefig(OUT/'interval_examples.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,g in df.groupby('model',sort=False): axes[0].plot(g.horizon,100*g.coverage,label=name); axes[1].plot(g.horizon,g.mean_width_c,label=name)
    axes[0].axhline(90,color='black',linestyle='--',label='nominal 90%'); axes[0].set(xlabel='Horizon (10-min steps)',ylabel='Coverage (%)'); axes[1].set(xlabel='Horizon (10-min steps)',ylabel='Average width (°C)')
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'coverage_width.png',dpi=160); plt.close(fig)
    print(df[df.horizon.isin([1,6,12,24])].to_string(index=False))

if __name__=='__main__': main()
