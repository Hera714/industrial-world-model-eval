"""总说明：LLM 生成意图、确定性校验、执行与审核；每次会话独立保存可追溯证据。"""
from pathlib import Path
from datetime import datetime,timezone
import json,uuid
import yaml
from benchmark.artifacts import atomic_write
from agents.llm import generate_with_llm, SYSTEM_PROMPT
from agents.tools import invoke
from agents.errors import describe_error

ROOT=Path(__file__).resolve().parents[1]

def run_agent(request, *, endpoint, model, generator=generate_with_llm):
    """异常原样记录并返回失败；工具执行前后写入同一个会话目录。"""
    folder=ROOT/'reports/week8/sessions'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:10])
    folder.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',request=request,events=[])
    def save():atomic_write(folder/'session.json',json.dumps(state,ensure_ascii=False,indent=2))
    def call(name,args):
        result=invoke(name,args)
        state['events'].append(dict(tool=name,result=result))
        save()
        if not result['ok']:raise RuntimeError(name+': '+result['error']['message'])
        return result['result']
    save()
    try:
        generated=generator(request,endpoint=endpoint,model=model)
        state.update(intent=generated['intent'],model_metadata=generated['metadata'],system_prompt=SYSTEM_PROMPT)
        config=generated['config']
        atomic_write(folder/'config.yaml',yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
        save()
        call('validate_config',{'config':config})
        # 普通流水线没有交互式审批；较大模型组合必须转交 LangGraph 审批节点。
        policy=yaml.safe_load((ROOT/'configs/agent/policy.yaml').read_text())
        if len(config['model']['instances'])>policy['approval_required_above_models']:
            raise ValueError('APPROVAL_REQUIRED: 模型数量超过阈值，请使用 LangGraph 人工审批')
        result=call('run_benchmark',{'config':config})
        # 保存项目相对路径，使挂载目录在宿主机或容器中变化时仍可定位同一 run。
        state['run_directory_relative']=Path(result['run_directory']).resolve().relative_to(ROOT).as_posix()
        save()
        audit=call('summarize_results',{'run_directory':result['run_directory']})
        state['audit']=audit
        if not audit['passed']:raise RuntimeError('AUDIT_FAILED: 结果未通过审核')
        state['status']='completed'
        atomic_write(folder/'report.md','# Agent 评测报告\n\n运行目录：`'+result['run_directory']+'`\n\n确定性审核通过；指标见该目录的 metrics.json 和 report.md。\n')
    except Exception as error:
        state.update(status='failed',error=describe_error(error))
    save()
    return folder,state
