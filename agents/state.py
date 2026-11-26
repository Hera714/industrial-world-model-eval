"""总说明：定义 LangGraph 各节点共享、可持久化的会话状态字段。"""
from typing import TypedDict


class AgentState(TypedDict, total=False):
    """保存用户请求、候选配置、事件、审批摘要及审核结果。"""
    request: str
    endpoint: str
    model: str
    require_approval: bool
    config: dict
    metadata: dict
    events: list
    error: dict
    run_directory: str
    audit: dict
    status: str
    approved_hash: str
