# Industrial World Model Evaluation

BOPTEST 世界模型预测与控制评测项目。

启动本地 BOPTEST 服务后，在项目根目录运行：

```bash
conda activate mvp
python run_benchmark.py --config configs/benchmark/boptest_smoke.yaml
```

每次运行的配置、轨迹、指标和报告保存在终端显示的 `reports/runs/` 独立目录。配置和使用说明见 [benchmark 协议](docs/benchmark_protocol.md)。

## 评测容器

先按现有 BOPTEST 部署启动 v0.9.0 的 web、worker 和依赖服务，保持 `project1-boptest_default` 网络可用。然后在项目根目录执行：

```bash
docker compose -f deployment/benchmark/compose.yaml build
docker compose -f deployment/benchmark/compose.yaml run --rm benchmark
```

项目目录挂载到容器 `/workspace`：配置、检查点、数据、FMU 校验文件必须已在宿主机准备好，结果写入宿主机 `reports/runs/`。容器使用 CPU 版 PyTorch，BOPTEST 服务单独运行。

## 回归和 Week 7 验收

将路径替换为本次实际 smoke run：

```bash
python -m benchmark.regression --run reports/runs/你的运行目录
python scripts/check_week7.py --run reports/runs/你的运行目录
```

加 `--integration` 会另行连接 BOPTEST 运行真实短回合。新环境需先准备清单中冻结的数据、模型和 FMU；文件缺失或哈希不符时离线检查会报错。详情见 [Week 7 报告](reports/week7/week7.md)。

## 失败后续跑

从本功能启用后产生的 run 续跑：

```bash
python run_benchmark.py --config configs/benchmark/boptest_smoke.yaml --resume-from reports/runs/原运行目录
```

续跑会创建新的 run，原 run 的失败上下文和部分轨迹保留。启动前核对完整配置、配置哈希及所有输入文件哈希；只复用带完整标记且哈希正确的 episode。中断时未完成的 episode 从初始化重新运行，避免重复推进可能已执行的 BOPTEST 请求。开环和排序重新计算；完成的在线采集及闭环 episode 可以复用。

旧版 run 没有 episode 检查点，不能跳过其中已完成的回合。当前不比较代码或依赖版本哈希，变更代码或运行环境后不应混合复用结果；强制中止的服务实例可能需要手动清理。

## GitHub Actions

提交并推送后，`.github/workflows/ci.yml` 在 push、pull request 或手动触发时运行 Python 静态检查和不依赖实验资产的短单元测试。真实 BOPTEST 测试需先部署服务与冻结数据，在 WSL 中执行：

```bash
BOPTEST_INTEGRATION=1 python -m unittest tests.test_benchmark_integration
```

CI 工作流和所需源码必须提交到 GitHub 后，才能在仓库的 Actions 页面查看结果。

## P2 批量执行

```bash
python scripts/batch_benchmark.py --config configs/benchmark/boptest_batch_smoke.yaml --output reports/week7/my_batch --workers 2
```

当前入口展开闭环/鲁棒性模型×日期×条件矩阵，支持 1～16 个独立进程；每个进程独立选择 BOPTEST testid，且写入独立任务目录。开环和策略排序使用原统一 Runner。批量模式不重复运行 embedded 基线。

同一命令再次运行会校验完整配置、实际数据/模型/模拟资产及源码指纹，再校验完成结果的所有文件哈希，仅执行尚未完成或失败的任务。配置或输入变化须使用新输出目录；损坏完成结果会报错，不能静默复用。每次尝试保留 attempt 日志、state.json、配置和独立 run；失败可按 identity 定位。

独占锁由协调器及已启动子进程继承；协调器被中断后须等待这些子进程结束才能重启，防止重复推进。尚未写 completed 状态的任务视为未完成，重启时重新初始化执行；不会接续一个未知的服务时间步。

真实验收：2 worker/4任务的实际峰值并发为2，testid全部独立；SIGTERM协调器后恢复保留已完成任务，全部4项完成。证据见 reports/week7/batch_verification.json。离线完整性测试：python -m unittest tests.test_batch_benchmark。
