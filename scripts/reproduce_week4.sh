#!/usr/bin/env bash
# 总说明：从项目根目录依序复现 Week 4，任何阶段失败立即停止。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run_dir="reports/week4/results/reproduction/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run_dir"
# 记录真实执行版本及工作区，便于识别未提交代码。
git rev-parse HEAD > "$run_dir/commit.txt"
git status --short > "$run_dir/worktree.txt"
git diff --binary > "$run_dir/tracked_changes.patch"
python -m pip freeze > "$run_dir/requirements.txt"
for script in collect_week4 train_week4 evaluate_week4 plot_week4 run_week4_control; do
    python "scripts/$script.py" 2>&1 | tee "$run_dir/$script.log"
done
printf 'Week 4 reproduction completed. Logs: %s\n' "$run_dir"
