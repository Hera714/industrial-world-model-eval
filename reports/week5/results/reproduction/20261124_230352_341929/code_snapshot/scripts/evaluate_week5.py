"""Week 5 统一入口：单步、长时 rollout，或一键统计/绘图/报告与追溯验证。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluators.prediction import evaluate_one_step
from evaluators.rollout import evaluate_week5_rollouts


def main():
    """只加载已有检查点，不在测试集训练或选参。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiments/week5_evaluation.yaml')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--stage', choices=['one_step', 'rollout', 'all', 'reproduce'], default='one_step')
    args = parser.parse_args()
    if args.stage == 'reproduce':
        from scripts.week5_pipeline import reproduce
        reproduce(ROOT/args.config,device=args.device)
        return
    if args.stage in ('one_step', 'all'):
        evaluate_one_step(ROOT/args.config, root=ROOT, device=args.device)
    if args.stage in ('rollout', 'all'):
        evaluate_week5_rollouts(ROOT/args.config, root=ROOT, device=args.device)


if __name__ == '__main__':
    main()
