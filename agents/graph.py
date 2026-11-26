"""总说明：LangGraph 真实状态机；需要批准时使用持久化 interrupt，所有工具仍由确定性代码执行。"""
import hashlib,json
from pathlib import Path
import yaml
from langgraph.graph import StateGraph,START,END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver
from agents.tools import invoke
from agents.llm import generate_with_llm
from agents.state import AgentState

def build_graph(generator=generate_with_llm,tool=invoke,checkpointer=None):
    """节点分离生成、校验、审批、执行、审核和总结；测试可注入替身。"""
    policy=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/agent/policy.yaml').read_text())
    limit=policy['approval_required_above_models']
    if not isinstance(limit,int) or isinstance(limit,bool) or limit<0:
        raise ValueError('人工审批模型数阈值非法')
    def requires_approval(state):
        # 不由调用方单独决定：超过冻结模型数量阈值时始终请求人工批准。
        instances=state['config'].get('model',{}).get('instances',[])
        return bool(state.get('require_approval') or len(instances)>limit)
    def generate(state):
        try:
            output=generator(state['request'],endpoint=state['endpoint'],model=state['model'])
            return dict(config=output['config'],metadata=output['metadata'],events=[dict(node='generate',intent=output['intent'])])
        except Exception as exc:return dict(error=dict(node='generate',message=str(exc)),status='failed')
    def validate(state):
        result=tool('validate_config',{'config':state['config']})
        update=dict(events=state['events']+[dict(node='validate',result=result)])
        if not result['ok']:update.update(error=result['error'],status='failed')
        return update
    def digest(config):
        return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    def approval(state):
        if not requires_approval(state):return {}
        fingerprint=digest(state['config'])
        answer=interrupt(dict(message='批准执行此配置',config=state['config'],config_sha256=fingerprint))
        if not isinstance(answer,dict) or answer.get('approved') is not True or answer.get('config_sha256')!=fingerprint:
            return dict(status='rejected',error=dict(node='approval',message='未批准或配置摘要不匹配'))
        return dict(approved_hash=fingerprint,events=state['events']+[dict(node='approval',approved=True,config_sha256=fingerprint)])
    def execute(state):
        if requires_approval(state) and state.get('approved_hash')!=digest(state['config']):
            return dict(status='failed',error=dict(node='execute',message='审批后配置已变化'))
        # 审批摘要只从图的受控状态传递，工具参数 Schema 不允许模型提供它。
        if tool is invoke:
            result=tool('run_benchmark',{'config':state['config']},approved_config_hash=state.get('approved_hash'))
        else:
            result=tool('run_benchmark',{'config':state['config']})
        update=dict(events=state['events']+[dict(node='execute',result=result)])
        if result['ok']:update['run_directory']=result['result']['run_directory']
        else:update.update(error=result['error'],status='failed')
        return update
    def audit(state):
        result=tool('summarize_results',{'run_directory':state['run_directory']})
        update=dict(events=state['events']+[dict(node='audit',result=result)])
        if not result['ok'] or not result['result']['passed']:update.update(error=dict(node='audit',result=result),status='failed')
        else:update['audit']=result['result']
        return update
    def summary(state):return dict(status='completed',events=state['events']+[dict(node='summary',message='审核通过，指标见原始 run')])
    graph=StateGraph(AgentState)
    for name,node in [('generate',generate),('validate',validate),('approval',approval),('execute',execute),('audit',audit),('summary',summary)]:graph.add_node(name,node)
    graph.add_edge(START,'generate')
    for node,next_node in [('generate','validate'),('validate','approval'),('approval','execute'),('execute','audit'),('audit','summary')]:
        graph.add_conditional_edges(node,lambda state:END if state.get('status') in ('failed','rejected') else 'next',{'next':next_node,END:END})
    graph.add_edge('summary',END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
