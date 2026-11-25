#!/usr/bin/env bash
# 总说明：重现固定动作响应与测试工况分解；复用冻结数据和检查点，不重新训练。
set -euo pipefail
cd "$(dirname "$0")/.."
week5_python="${WEEK5_PYTHON:-python}"
"$week5_python" scripts/action_week5.py
"$week5_python" scripts/decompose_week5.py
"$week5_python" scripts/plot_week5_diagnostics.py
"$week5_python" -m unittest tests.test_week5_diagnostics
"$week5_python" scripts/check_week5_diagnostics.py
