#!/usr/bin/env bash
# 总说明：独立复现验证集校准和测试集固定区间评价，不改写主协议结果。
set -euo pipefail
cd "$(dirname "$0")/.."
week5_python="${WEEK5_PYTHON:-python}"
# 先将验证集参数持久化；测试阶段拒绝配置、参数或相关代码被悄悄更改。
"$week5_python" scripts/calibrate_week5.py --phase calibrate
"$week5_python" scripts/calibrate_week5.py --phase evaluate
"$week5_python" scripts/plot_week5_calibration.py
"$week5_python" -m unittest tests.test_calibration
"$week5_python" scripts/check_week5_calibration.py
