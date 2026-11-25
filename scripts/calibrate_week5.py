"""Week 5 概率校准入口：验证集拟合与测试集评价可分开执行，避免测试时调参。"""
from pathlib import Path
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluators.calibration import calibrate,evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/experiments/week5_calibration.yaml')
    parser.add_argument('--phase',choices=['calibrate','evaluate','all'],default='all')
    args=parser.parse_args()
    path=ROOT/args.config
    if args.phase in ('calibrate','all'): calibrate(path,ROOT)
    if args.phase in ('evaluate','all'):
        result=evaluate(path,ROOT)
        print(result.query('mode == "free_running" and condition == "recorded_observation" and method == "validation_calibrated" and horizon in [1,20,80]')[['family','variable','horizon','coverage_mean','mean_width_c']].to_string(index=False))


if __name__=='__main__': main()
