#!/usr/bin/env bash
# 总说明：复现 Week 5 已冻结主实验和扩展；不重新训练、不重新冻结资产。
set -euo pipefail
cd "$(dirname "$0")/.."
# 先生成原始模型与独立投影模型输出，再复算投影前后诊断。
python scripts/evaluate_week5.py --stage all --device cpu
python scripts/evaluate_week5.py --stage rollout --config configs/experiments/week5_projection.yaml --device cpu
python scripts/check_week5_projection.py
# IB 使用独立信息集与原生尺度，不混入热工主表。
python scripts/evaluate_week5_ib.py
python scripts/check_week5_extensions.py
python scripts/check_week5_rollout.py
# 可选补充区间：按完整 episode 分组重采样，固定训练种子集合。
python scripts/bootstrap_week5.py
python scripts/bootstrap_week5.py --folder reports/week5/results/projection/rollout
python scripts/bootstrap_week5.py --folder reports/week5/results/ib_history
python -m unittest tests.test_week5_extensions tests.test_validity tests.test_week5_rollout tests.test_week5_prediction tests.test_metrics tests.test_dynamics_api tests.test_physics_constraints
