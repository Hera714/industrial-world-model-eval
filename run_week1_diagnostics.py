"""P1：训练集时序诊断及固定 Persistence 的验证残差预测区间。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.split import split_by_time

OUT = Path('reports/week1_p1')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv('data/raw/energydata_complete.csv')
    splits = split_by_time(frame); train = splits['train']
    y = train.T1.to_numpy(dtype=float)
    hourly = train.assign(hour=train.date.dt.hour + train.date.dt.minute / 60).groupby('hour').T1.mean()
    centered = y-y.mean(); denom = np.dot(centered, centered)
    acf = np.array([np.dot(centered[:-lag], centered[lag:])/denom if lag else 1.0 for lag in range(289)])
    q1, q3 = np.quantile(y, [.25, .75]); low, high = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
    stats = {k: {'count':len(v), 'mean':float(v.T1.mean()), 'std':float(v.T1.std()), 'median':float(v.T1.median()), 'iqr_outlier_count_train_threshold':int(((v.T1<low)|(v.T1>high)).sum())} for k,v in splits.items()}
    diff = pd.to_datetime(frame.date).sort_values().diff().dropna()
    info = {'t1_split_stats':stats, 'train_iqr_bounds':[float(low),float(high)], 'sampling_interval_counts':{str(k):int(v) for k,v in diff.value_counts().items()}, 'acf_lag_1':float(acf[1]), 'acf_lag_144':float(acf[144])}
    (OUT/'diagnostics.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    fig, ax = plt.subplots(2,2,figsize=(12,8)); ax[0,0].plot(train.date, y, linewidth=.5); ax[0,0].set_title('Training T1 over time'); ax[0,0].set_ylabel('°C')
    # 减少日期刻度并倾斜标签，避免横坐标互相遮挡。
    import matplotlib.dates as mdates
    ax[0,0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax[0,0].tick_params(axis='x', labelrotation=30, labelsize=8)
    for label in ax[0,0].get_xticklabels(): label.set_horizontalalignment('right')
    ax[0,1].plot(hourly.index,hourly.values); ax[0,1].set(xlabel='Hour of day',ylabel='Mean T1 (°C)',title='Training daily profile')
    ax[1,0].plot(np.arange(289)/6,acf); ax[1,0].set(xlabel='Lag (hours)',ylabel='ACF',title='Training autocorrelation')
    ax[1,1].boxplot([splits[k].T1 for k in ('train','val','test')],tick_labels=['train','val','test'],showfliers=False); ax[1,1].set(ylabel='T1 (°C)',title='Split distributions (diagnostic only)')
    fig.tight_layout(); fig.savefig(OUT/'diagnostics.png',dpi=150); plt.close(fig)
    meta=json.loads(Path('data/processed/week1/metadata.json').read_text()); mu=meta['scaler']['mean']['T1']; scale=meta['scaler']['scale']['T1']; idx=meta['features'].index('T1')
    with np.load('data/processed/week1/val.npz') as f: Xv=f['X']; yv=f['y']*scale+mu
    with np.load('data/processed/week1/test.npz') as f: Xt=f['X']; yt=f['y']*scale+mu; starts=f['target_start']
    pv=np.repeat((Xv[:,-1,idx]*scale+mu)[:,None],yv.shape[1],axis=1); pt=np.repeat((Xt[:,-1,idx]*scale+mu)[:,None],yt.shape[1],axis=1)
    # 固定模型；逐预测步分别拟合 5% 和 95% 验证残差分位数。
    lower, upper = np.quantile(yv-pv,[.05,.95],axis=0); lo=pt+lower; hi=pt+upper
    coverage=((yt>=lo)&(yt<=hi)).mean(axis=0); width=(hi-lo).mean(axis=0)
    table=pd.DataFrame({'horizon':np.arange(1,yt.shape[1]+1),'minutes':np.arange(1,yt.shape[1]+1)*10,'coverage':coverage,'mean_width_c':width,'residual_q05_c':lower,'residual_q95_c':upper})
    table.to_csv(OUT/'interval_metrics.csv',index=False)
    np.savez_compressed(OUT/'interval_predictions.npz',actual=yt,predicted=pt,lower=lo,upper=hi,target_start=starts)
    fig,ax=plt.subplots(figsize=(9,4)); h=np.arange(1,yt.shape[1]+1); ax.plot(h,yt[0],label='actual'); ax.plot(h,pt[0],label='persistence'); ax.fill_between(h,lo[0],hi[0],alpha=.25,label='nominal 90% interval'); ax.set(xlabel='Horizon (10-min steps)',ylabel='T1 (°C)',title='Test window 0'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'interval_example.png',dpi=150); plt.close(fig)
    print(json.dumps(info,indent=2)); print(table.iloc[[0,5,11,23]].to_string(index=False))

if __name__ == '__main__': main()
