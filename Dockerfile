# 总说明：只安装评测运行依赖；代码和冻结实验资产由 Compose 挂载。
FROM python:3.11-slim-bookworm@sha256:ef56ae98435cc40d8a27020b0b32a9941515acd5840f0fba34858bb34dd362d1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
RUN git config --global --add safe.directory /workspace
COPY deployment/benchmark/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir -r /tmp/requirements.lock
ENV PYTHONUNBUFFERED=1
CMD ["python", "run_benchmark.py", "--config", "configs/benchmark/boptest_smoke_container.yaml"]
