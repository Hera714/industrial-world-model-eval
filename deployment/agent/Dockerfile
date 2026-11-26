# 总说明：在 Week 7 固定评测镜像基础上增加本地 LLM 推理依赖，模型资产由外部挂载。
FROM benchmark-benchmark:latest
COPY deployment/agent/requirements.lock /tmp/agent-requirements.lock
RUN python -m pip install --no-cache-dir -r /tmp/agent-requirements.lock
