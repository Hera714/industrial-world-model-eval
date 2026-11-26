# Industrial World Model Evaluation

BOPTEST 世界模型预测与控制评测项目。

本项目研究一个问题：模型能否根据当前观测、控制动作和天气预测楼宇未来状态，以及预测误差是否能反映闭环控制表现。主要平台是 BOPTEST 的 `bestest_hydronic_heat_pump` 仿真案例；结果不是实际建筑部署结果。

| 想看什么 | 位置 |
| --- | --- |
| 架构和模块职责 | [架构图](docs/architecture.md) |
| Week 5–7 代表性实验表、失败案例和局限 | [Week 8 报告](reports/week8/week8.md) |
| 5 分钟介绍、10 分钟深挖、10 个问答 | [面试材料](docs/interview/) |
| Agent 工作边界和人工审批 | [Agent 协议](docs/agent_protocol.md) |
| 数据、第三方代码来源 | [第三方来源与许可线索](docs/third_party.md) |
| 实验文件如何保存 | [产物保存策略](docs/artifact_policy.md) |

3 分钟体验的前提是：已准备好 BOPTEST 服务、冻结数据和检查点、Qwen 模型及镜像；首次安装、构建和下载不计入。CPU 容器中一次已准备资产后的实测为 92.96 秒，命令见下文“在 CPU 容器中复现”；运行时间因设备和服务负载而异。[从零准备与故障排查](docs/setup_reproduction.md)列出了前置步骤。

## 环境准备

使用 Python 3.11。现有 `mvp` 环境如已安装项目依赖及可用的 PyTorch，可在项目根目录执行 `conda activate mvp`；依赖版本见 `pyproject.toml`，CPU 评测容器的依赖清单见 `deployment/benchmark/requirements.lock`。本地 Agent 还需 `transformers==4.57.3` 和 `accelerate==1.12.0`；LangGraph 审批另需 `langgraph==0.6.11` 和 `langgraph-checkpoint-sqlite==3.1.3`。如已有 CUDA 版 PyTorch，安装其他依赖时请核对 pip 的安装计划，避免无意替换。

BOPTEST 需作为独立服务启动，案例、步长和版本见 `configs/environments/boptest.yaml`、[Week 6 协议](docs/week6_protocol.md)。历史本机部署使用 BOPTEST v0.9.0；模型文件和冻结数据不随镜像自动生成，需按配置清单准备并通过哈希校验。Agent 本地模型的版本与文件哈希见 `configs/agent/local_model_manifest.json`。已有服务可先检查 `http://127.0.0.1/testcases` 是否响应，再运行下方命令。

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

## MLflow 本地追踪

安装可选客户端（不属于默认 CPU 容器依赖）：

```bash
python -m pip install mlflow-skinny==2.22.5
python run_benchmark.py --config configs/benchmark/boptest_smoke.yaml --tracking-config configs/tracking/local.yaml
```

`reports/mlruns/` 保存本地 MLflow 记录，每个 benchmark 对应一个 MLflow run。参数包含模型检查点哈希和种子；逐变量误差、控制指标存为可查询 metric，完整指标和配置另作 artifact。`tracking.json` 提供 MLflow run ID 和 tracking URI；MLflow 标签 `benchmark.directory` 指向原始轨迹目录。大型原始轨迹留在原 run 目录，不重复复制，哈希清单用于核验。

查询最近记录：

```python
from pathlib import Path
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri=Path('reports/mlruns').resolve().as_uri())
experiment = client.get_experiment_by_name('industrial-world-model-eval')
for run in client.search_runs([experiment.experiment_id], order_by=['start_time DESC']):
    print(run.info.run_id, run.info.status, run.data.tags.get('benchmark.directory'))
```

成功为 FINISHED，异常为 FAILED；失败上下文也上传为 artifact。追踪同步失败不会覆盖原 benchmark 异常，会留下 `tracking_error.json` 并发出警告；强制杀进程仍可能留下 RUNNING，不能视为完成。需要网页 UI 时可另外安装相同版本完整 `mlflow` 并运行 `mlflow ui --backend-store-uri ./reports/mlruns`。

## Week 8 Agent（开发中）

预备：本地 BOPTEST、冻结资产和可用的 OpenAI 兼容模型服务。模型地址及模型名由用户明确配置，密钥可放在 `AGENT_LLM_API_KEY` 环境变量，不写入配置或日志。安装状态机可选依赖：

```bash
python -m pip install 'langgraph==0.6.11' 'langgraph-checkpoint-sqlite==3.1.3'
python -m unittest tests.test_agent_tools tests.test_agent_pipeline tests.test_agent_errors tests.test_agent_graph tests.test_agent_graph_sqlite
```

当已有可用模型服务时，使用 `scripts/run_agent.py --request <任务> --endpoint <chat_completions地址> --model <模型名>`。当前 Agent 工具仅允许选择冻结 smoke 模板的模型；其他配置会被拒绝。本地 Qwen 已完成真实端到端和 CPU 容器复现；外部模型服务入口尚未做同等真实验收。详见 `docs/agent_protocol.md`。

### 本地 LLM 快速体验（模型和 BOPTEST 已准备）

下面命令需要完成约 2.9 GiB 模型权重下载，准备冻结 tokenizer 文件并启动 BOPTEST；首次准备和下载不计入三分钟。安装 `transformers==4.57.3`、`accelerate==1.12.0` 后：

```bash
python scripts/run_local_agent.py \
  --model-dir external/qwen-agent \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --expected-sha256 dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee \
  --request '运行 BOPTEST，比较 persistence 和 arx'
```

模型及元数据需符合 `configs/agent/local_model_manifest.json`，入口会先验证哈希。一次真实成功案例见 `reports/week8/sessions/20261125T161119997809Z_c54402617b/`。已准备资产后，隔离 CPU 容器中的一次计时运行用时 92.96 秒；不含镜像构建与资产下载，其他硬件或服务负载下时间可能不同。遇到服务超时请检查服务状态，勿反复请求造成实例积压。

### 在 CPU 容器中复现

前提：已准备冻结数据、检查点和 `external/qwen-agent` 模型，BOPTEST 已启动并可从主机网络访问 `127.0.0.1`。Docker Desktop 应开启 host networking 支持；首次构建及下载不计入运行时间。以下在 WSL 项目根目录执行：

```bash
docker compose -f deployment/benchmark/compose.yaml build
docker build -f deployment/agent/Dockerfile -t industrial-agent-week8:local .
docker run --rm --network host --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=4 -e HF_HUB_OFFLINE=1 \
  -v "$PWD:/workspace" -w /workspace industrial-agent-week8:local \
  python scripts/run_local_agent.py --device cpu --model-dir external/qwen-agent \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --expected-sha256 dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee \
  --request '运行 BOPTEST，比较 persistence 和 arx'
```

运行输出包含 session 路径。回到宿主机后将路径中的 `/workspace/` 替换为项目根目录，执行 `python scripts/check_agent_session.py --session <session目录>` 核对。此容器使用 CPU，速度随硬件而变；本机真实通过案例及复核记录见 `reports/week8/container_timed_verification.json`。Agent 镜像以先构建的 `benchmark-benchmark:latest` 为基础，新增依赖版本见 `deployment/agent/requirements.lock`。
