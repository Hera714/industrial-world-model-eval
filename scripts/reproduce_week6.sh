#!/usr/bin/env bash
# 总说明：从统一配置运行十一模型完整复现，逐步失败日志由Python入口保存。
set -euo pipefail
cd "$(dirname "$0")/.."
week6_python="${WEEK6_PYTHON:-python}"
"$week6_python" scripts/evaluate_week6.py --config configs/experiments/week6_pipeline.yaml --stage reproduce
