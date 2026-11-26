# 从零准备 BOPTEST 与 Agent 演示

本文面向 Linux / WSL 2。下面区分“资产准备”和“开始评测”：README 的三分钟计时只指资产、镜像和服务都已就绪之后的一次运行，不包含下载和构建。BOPTEST 是仿真平台，硬件与网络状态会影响时间。

## 1. 下载项目与准备 Python

```bash
git clone https://github.com/Hera714/industrial-world-model-eval.git
cd industrial-world-model-eval
```

准备 Python 3.11 和可用的 Docker Desktop / Docker Engine 与 Compose。下面是独立 CPU 环境的例子，不会修改已有 `mvp` 环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r deployment/benchmark/requirements.lock
```

本地 Qwen Agent 另需 `python -m pip install -r deployment/agent/requirements.lock`；如需 LangGraph，另装 `python -m pip install 'langgraph==0.6.11' 'langgraph-checkpoint-sqlite==3.1.3'`。如使用现有 CUDA 版 PyTorch，请先检查 pip 将安装什么，避免 CPU 锁文件覆盖原有 GPU 版本。

## 2. 准备固定版本的 BOPTEST

在项目根目录执行；若目录已经存在，先检查版本和本地修改，不要重复 clone 或覆盖：

```bash
mkdir -p external
git clone --branch v0.9.0 --depth 1 https://github.com/ibpsa/project1-boptest.git external/project1-boptest
git -C external/project1-boptest rev-parse HEAD
```

预期提交为 `9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9`。固定资产和哈希记录在 `reports/week6/results/assets/manifest.json`。检查本地 FMU：

```bash
sha256sum external/project1-boptest/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu
```

预期 SHA256 为 `674b9500c1c89fdbec64f2c053af8ddcd06afebe63bd7c1b7fc708692a11863e`；不一致时先查明版本和文件来源，不要修改哈希绕过检查。

在 BOPTEST 目录尝试按其 Compose 配置启动：

```bash
cd external/project1-boptest
docker compose up -d web worker provision
docker compose ps -a
cd ../..
curl --noproxy '*' -fsS http://127.0.0.1/testcases
```

构建和案例初始化可能需要较长时间；可通过 `docker compose logs --tail 80 worker provision` 在 BOPTEST 目录查看进度。HTTP 响应只能说明服务可访问，不能单独证明案例可用或 FMU 版本正确。端口占用、Docker Hub 镜像变化、代理和 Ubuntu 包源都可能导致构建失败；若某步骤报错，先保留完整日志，不要持续重复提交案例选择请求。

本项目历史运行使用过本机构建修改，归档在 `deployment/boptest/local-build.patch`，说明见 `deployment/boptest/README.md`。该补丁还依赖未纳入仓库的 `apt-seed`、Miniconda 安装包和 MinIO 镜像；文件路径与哈希在资产清单中。**只 clone 和 apply 补丁并不保证能重建历史镜像。** 若需重建这套修改，先准备清单中的文件并核对哈希，再从项目根目录执行：

```bash
git -C external/project1-boptest apply --check ../../deployment/boptest/local-build.patch
git -C external/project1-boptest apply ../../deployment/boptest/local-build.patch
```

若官方镜像不可拉取，需要先根据上游当前镜像可用性或归档的 MinIO 源码信息处理；本仓库没有完整自动化脚本，不能保证某个镜像源始终可用。使用其他构建方式时，必须重新核实服务端实际加载的 FMU；项目提供 `python scripts/check_week6_deployed.py`，它会创建真实案例并依赖正常运行的 Docker 服务。

## 3. 检查数据、检查点与配置

Week 6 的已跟踪检查点、采集数据、冻结场景清单位于 `reports/week6/results/` 和 `configs/benchmark/manifests/`。先离线核验默认短评测的配置与文件，不会连接 BOPTEST：

```bash
python -c 'from pathlib import Path; import yaml; from benchmark.config import validate_config; c=yaml.safe_load(Path("configs/benchmark/boptest_smoke.yaml").read_text()); validate_config(c, Path.cwd()); print("配置和文件校验通过")'
```

若提示缺少 `external/project1-boptest/.../wrapped.fmu`，先完成第 2 节；若有其他文件缺失或 SHA256 不符，按报错路径检查清单与下载来源。不能仅因为本地服务可访问就跳过资产检查。

## 4. 下载本地 Qwen 资产（Agent 演示需要）

下面在项目根目录使用 Hugging Face 客户端下载**冻结版本所需文件**；需网络与足够磁盘空间。权重约 2.9 GiB，首次下载不计入演示时间。

```bash
python -m pip install -r deployment/agent/requirements.lock
python - <<'PY'
import json
from huggingface_hub import snapshot_download

with open('configs/agent/local_model_manifest.json', encoding='utf-8') as f:
    manifest = json.load(f)
snapshot_download(
    repo_id=manifest['repository'], revision=manifest['revision'],
    local_dir='external/qwen-agent',
    allow_patterns=[manifest['weight'], *manifest['metadata_sha256'].keys()],
)
PY
```

运行前校验所有文件：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path('configs/agent/local_model_manifest.json').read_text())
files = {**manifest['metadata_sha256'], manifest['weight']: manifest['weight_sha256']}
for name, expected in files.items():
    with (Path('external/qwen-agent') / name).open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'{name}: SHA256 不匹配')
    print(name, 'OK')
PY
```

网络不稳定时可参照 `scripts/download_model_ranges.py --help` 分段下载权重，但仍需分别准备并校验其他文件。

## 5. 运行与验收

确认服务和资产均准备好后，先执行无 Agent 的短评测：

```bash
python run_benchmark.py --config configs/benchmark/boptest_smoke.yaml
```

命令输出 run 目录；其中 `config.yaml`、`metrics.json`、`trajectories.csv`、`status.json`、`input_hashes.json` 和日志可追溯本次运行。可用 `python scripts/check_week7.py --run <实际run目录>` 检查。Agent CPU 容器和真实自然语言演示命令见 [README](../README.md#在-cpu-容器中复现)。

提交代码前先看 `git status`。新产生的本地 run、MLflow 和 Agent session 默认被忽略；确需公开某一次结果时，先检查内容，再有选择地 `git add -f <路径>`。已有已跟踪文件不会因为新增忽略规则自动退出 Git。
