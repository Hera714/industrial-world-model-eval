#!/usr/bin/env bash
# 总说明：残差MLP补充实验复现，原模型和冻结数据须已就位，真实闭环需要本地BOPTEST。
set -euo pipefail
cd "$(dirname "$0")/.."
week6_python="${WEEK6_PYTHON:-python}"
"$week6_python" scripts/check_week6_protocol.py
"$week6_python" scripts/extend_residual_week6.py
export WEEK6_MODELS_MANIFEST=reports/week6/results/residual_mlp/models.json
export WEEK6_OPEN_OUTPUT=reports/week6/results/residual_mlp/open_loop
"$week6_python" scripts/evaluate_week6_open.py
"$week6_python" scripts/run_week6_control.py
"$week6_python" scripts/run_week6_control.py --goal-shift
"$week6_python" scripts/run_week6_control.py --robustness
"$week6_python" scripts/finish_week6_residual.py
"$week6_python" scripts/rank_week6_residual.py
"$week6_python" scripts/plot_week6_residual.py
