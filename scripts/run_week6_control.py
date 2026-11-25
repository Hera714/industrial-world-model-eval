"""总说明：BOPTEST 真实闭环对照，重放同一初始上下文；只改模型，保持规划预算和独立舒适标准。"""
from pathlib import Path
import sys,json,time,hashlib,os
import numpy as np
import pandas as pd
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulators.boptest_control import BoptestControlEnv,measurement_state
from controllers.boptest_mpc import stage_cost
from controllers.mpc import create_boptest_mpc
from models.boptest_dynamics import BoptestHistoryModel,exogenous_features


from evaluators.closed_loop import run_boptest_episode as run

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--robustness',action='store_true');parser.add_argument('--goal-shift',action='store_true');parser.add_argument('--only-episodes',nargs='+');args=parser.parse_args()
    torch.set_num_threads(2);cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text())
    models=json.loads(Path(os.environ.get('WEEK6_MODELS_MANIFEST',ROOT/cfg['output']/'training/models.json')).read_text())['models']
    models=([dict(name='embedded',kind='embedded')] if not os.environ.get('WEEK6_MODELS_MANIFEST') else [])+models
    episodes=[e for e in cfg['episodes'] if e['split'] in ('id_test','ood_cold','ood_warm')]
    extension=yaml.safe_load((ROOT/'configs/experiments/week6_boptest_extensions.yaml').read_text())
    if args.goal_shift or args.robustness:episodes=[e for e in episodes if e['episode_id'] in extension['primary_reference_episodes']]
    if args.only_episodes:
        if not set(args.only_episodes).issubset({e['episode_id'] for e in episodes}):raise ValueError('Unknown/nonfrozen episode')
        episodes=[e for e in episodes if e['episode_id'] in args.only_episodes]
    for condition in (['goal_plus1'] if args.goal_shift else (['noise_0p3','missing_4'] if args.robustness else ['clean'])):
        for episode in episodes:
            for spec in models:run(cfg,episode,spec,condition)
    files=list((ROOT/cfg['output']/'closed_loop').glob('*.json'))
    metrics=[json.loads(p.read_text()) for p in files if not p.name.endswith(('_forecast.json','_failure.json'))]
    pd.DataFrame(metrics).to_csv(ROOT/cfg['output']/'closed_loop/metrics.csv',index=False)


if __name__=='__main__':main()
