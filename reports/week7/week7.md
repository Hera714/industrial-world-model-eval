# Week 7：自动化评测与复现

## 当前状态

统一配置、离线校验、模型加载、Runner、独立 run、短回合集成测试和 smoke 回归检查已实现。本地 BOPTEST 0.9.0 与 CPU 评测容器已完成真实 smoke；96 步历史上下文，8 步评分，两个模型与 embedded 基线。完整 Week6 矩阵未在本周重跑。

## 复现与追溯

入口和命令见项目 README。每次 run 的 `config.yaml` 记录模型、种子和时间块；`input_hashes.json` 记录数据与检查点哈希；`versions.json` 记录 commit、工作区状态和依赖版本。逐步轨迹、预测、指标和错误分别保存在 `trajectories.csv`、`predictions.npz`、`metrics.json` 和 `failure.json`。

`python scripts/check_week7.py --run <目录>` 执行离线检查；加 `--integration` 另行连接真实 BOPTEST。completed 表示流程完成，须结合有效比例和 fallback 阅读。

## 限制

新机器需自行准备 BOPTEST 服务、FMU、数据和检查点并通过哈希检查；尚未在完全新机器重建全部资产。Python 依赖固定版本但未固定包哈希，补充 tzdata 和 Git 后已重新构建并运行真实 smoke。运行时若 `git_dirty` 为 true，commit 不能唯一代表未提交代码，需保存代码快照或提交后重跑。MLflow 与 CI 留待 P1。

已验收的早期容器 run 因镜像未安装 Git，commit 为空，不能通过版本追溯验收；Dockerfile 已补装 Git，修复后的新镜像已重新构建，新的容器 smoke 已记录 commit 并通过 `check_week7.py`。宿主机历史 run 记录了 commit，但工作区有未提交改动。

## 失败后续跑

从本功能启用后产生的 run 续跑：

```bash
python run_benchmark.py --config configs/benchmark/boptest_smoke.yaml --resume-from reports/runs/原运行目录
```

续跑会创建新的 run，原 run 的失败上下文和部分轨迹保留。启动前核对完整配置、配置哈希及所有输入文件哈希；只复用带完整标记且哈希正确的 episode。中断时未完成的 episode 从初始化重新运行，避免重复推进可能已执行的 BOPTEST 请求。开环和排序重新计算；完成的在线采集及闭环 episode 可以复用。

旧版 run 没有 episode 检查点，不能跳过其中已完成的回合。当前不比较代码或依赖版本哈希，变更代码或运行环境后不应混合复用结果；强制中止的服务实例可能需要手动清理。

## P2 批量执行

```bash
python scripts/batch_benchmark.py --config configs/benchmark/boptest_batch_smoke.yaml --output reports/week7/my_batch --workers 2
```

当前入口展开闭环/鲁棒性模型×日期×条件矩阵，支持 1～16 个独立进程；每个进程独立选择 BOPTEST testid，且写入独立任务目录。开环和策略排序使用原统一 Runner。批量模式不重复运行 embedded 基线。

同一命令再次运行会校验完整配置、实际数据/模型/模拟资产及源码指纹，再校验完成结果的所有文件哈希，仅执行尚未完成或失败的任务。配置或输入变化须使用新输出目录；损坏完成结果会报错，不能静默复用。每次尝试保留 attempt 日志、state.json、配置和独立 run；失败可按 identity 定位。

独占锁由协调器及已启动子进程继承；协调器被中断后须等待这些子进程结束才能重启，防止重复推进。尚未写 completed 状态的任务视为未完成，重启时重新初始化执行；不会接续一个未知的服务时间步。

真实验收：2 worker/4任务的实际峰值并发为2，testid全部独立；SIGTERM协调器后恢复保留已完成任务，全部4项完成。证据见 reports/week7/batch_verification.json。离线完整性测试：python -m unittest tests.test_batch_benchmark。

批量最终验收补充：不可达服务的4个子任务均保存失败身份和上下文；重试后每项保留2份尝试日志。配置变化被拒绝复用。14项批量完整性、episode恢复和Runner回归测试通过。完整证据位于 `reports/week7/batch_verification.json`。
