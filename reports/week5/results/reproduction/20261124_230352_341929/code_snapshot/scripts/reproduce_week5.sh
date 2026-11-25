#!/usr/bin/env bash
# 总说明：使用一个冻结配置复现 Week 5 热工主评测、表格、图、报告及验收记录。
set -euo pipefail
cd "$(dirname "$0")/.."
# 使用当前已激活的 mvp Python；必要时通过 WEEK5_PYTHON 指定解释器。
week5_python="${WEEK5_PYTHON:-python}"
week5_config="${1:-configs/experiments/week5_evaluation.yaml}"
week5_device="${2:-cpu}"
"$week5_python" scripts/evaluate_week5.py --config "$week5_config" --device "$week5_device" --stage reproduce
