#!/usr/bin/env bash
# 总说明：评分已冻结的三成员测试预测，不训练、不调整区间，也不假设正态 NLL。
set -euo pipefail
cd "$(dirname "$0")/.."
week5_python="${WEEK5_PYTHON:-python}"
"$week5_python" scripts/score_week5_probability.py
"$week5_python" scripts/plot_week5_probability.py
"$week5_python" -m unittest tests.test_probabilistic tests.test_summary
"$week5_python" scripts/check_week5_probability.py
