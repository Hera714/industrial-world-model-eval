"""从逐 episode 指标重建两层汇总与配对差值；不重新训练或推理。"""
from pathlib import Path
import sys
import argparse
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluators.summary import summarize_episode_metrics,episode_seed_metrics,paired_episode_differences,summarize_pairs


def summarize_folder(folder):
    """同一文件夹只允许一种环境，明确区分 episode SD 与训练种子 SD。"""
    metrics=pd.read_csv(folder/'metrics_by_episode.csv')
    if 'unit' not in metrics: metrics['unit']='Celsius'
    if metrics.unit.nunique()!=1: raise ValueError('Do not pool different environments or units')
    summary,seeds=summarize_episode_metrics(metrics)
    episodes=episode_seed_metrics(metrics)
    if 'ib_gru' in set(metrics.family):
        information={'ib_persistence':'current','ib_mlp':'current','ib_history_mlp':'history20','ib_gru':'history20'}
    else:
        information={family:'thermal_full_state' for family in metrics.family.unique()}
    pairs=paired_episode_differences(episodes,information)
    pair_summary=summarize_pairs(pairs)
    # 避免改变原指标含义：新汇总与既有结果逐项一致后再替换汇总表。
    prior=pd.read_csv(folder/'metrics_by_horizon.csv')
    keys=['model','mode','condition','horizon','definition','variable']
    comparison=summary.merge(prior,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    import numpy as np
    for metric in ('mae_c_mean','rmse_c_mean','mae_native_mean','rmse_native_mean'):
        if metric+'_new' in comparison:
            np.testing.assert_allclose(comparison[metric+'_new'],comparison[metric+'_old'],equal_nan=True)
    summary.to_csv(folder/'metrics_by_horizon.csv',index=False)
    summary[summary.horizon.isin([1,5,10,20,40,80])].to_csv(folder/'metrics_key_horizons.csv',index=False)
    seeds.to_csv(folder/'metrics_across_seeds.csv',index=False)
    episodes.to_csv(folder/'metrics_by_episode_across_seeds.csv',index=False)
    pairs.to_csv(folder/'paired_episode_differences.csv.gz',index=False,compression='gzip')
    pair_summary.to_csv(folder/'paired_summary.csv',index=False)
    pair_summary[pair_summary.horizon.isin([1,5,10,20,40,80])].to_csv(folder/'paired_key_horizons.csv',index=False)
    (folder/'summary_protocol.json').write_text(json.dumps(dict(
        independent_unit='episode; overlapping windows aggregated before this module',
        main_estimand='equal-weight episode mean per model run, then fixed training-seed mean and sample SD',
        pairing='family seed means within identical episode/context; delta=A-B; negative favours A',
        failed_pairs='retain outer-joined rows; delta missing unless all runs have complete finite support and no failure',
        information_groups=information,seed_sd='sample SD; undefined for one run',
        units=metrics.unit.unique().tolist(),confidence_intervals='existing supplementary episode bootstrap not recalculated here'),indent=2))
    print(f'{folder}: {len(episodes)} episode/family rows; {len(pairs)} paired episode rows; error aggregates unchanged',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',action='append')
    args=parser.parse_args()
    for folder in args.folder or ['reports/week5/results/rollout','reports/week5/results/projection/rollout','reports/week5/results/ib_history']:
        summarize_folder(ROOT/folder)


if __name__=='__main__': main()
