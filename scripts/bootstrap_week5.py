"""可选95%置信区间：episode 分组重采样，所有模型使用相同回合抽样索引。"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',default='reports/week5/results/rollout')
    parser.add_argument('--replicates',type=int,default=2000)
    args=parser.parse_args()
    folder=ROOT/args.folder
    frame=pd.read_csv(folder/'metrics_by_episode.csv')
    frame=frame[frame.horizon.isin([1,5,10,20,40,80]) & frame.definition.isin(['terminal','prefix_complete'])]
    names=['family','mode','condition','horizon','definition','variable']
    ids=sorted(frame.episode_id.unique()); rng=np.random.default_rng(2026)
    samples=rng.integers(0,len(ids),(args.replicates,len(ids)))
    rows=[]
    columns=[c for c in ('mae_c','rmse_c','mae_native','rmse_native') if c in frame]
    for key,group in frame.groupby(names):
        # 固定训练种子集合，先在每个回合内平均；不把训练种子伪装成额外独立回合。
        ep=group.groupby('episode_id')[columns].mean().reindex(ids)
        for metric in columns:
            values=ep[metric].to_numpy()
            if not np.isfinite(values).all() or not np.isfinite(group[metric]).all():
                rows.append(dict(zip(names,key))|dict(metric=metric,estimate=np.nan,low=np.nan,high=np.nan,
                    episodes=len(ids),replicates=args.replicates,reason='Missing episode error; interval not reported'))
                continue
            bootstrap=values[samples].mean(axis=1)
            low,high=np.quantile(bootstrap,[.025,.975])
            rows.append(dict(zip(names,key))|dict(metric=metric,estimate=values.mean(),low=low,high=high,
                episodes=len(ids),replicates=args.replicates,reason='Fixed training seeds; episode percentile bootstrap'))
    pd.DataFrame(rows).to_csv(folder/'episode_bootstrap_ci.csv',index=False)
    print('Saved paired episode-bootstrap intervals:',folder)


if __name__=='__main__': main()
