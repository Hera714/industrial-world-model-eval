"""总说明：OpenAI 兼容模型接口；密钥只从环境变量读取，输出结构化意图后交由确定性校验。"""
import json,os
from urllib.request import Request,urlopen
from agents.response_parser import parse_response
from agents.intent_guard import require_explicit_boptest

PROMPT_VERSION='week8-intent-v1'
SYSTEM_PROMPT=('你是楼宇评测配置助手。只输出 JSON 对象，字段严格为 environment 和 models。'
 'environment 只能是 boptest；models 是用户明确指定的检查点名称列表。'
 '无法确定或要求额外功能时返回 {"error":"说明原因"}，不得自行猜测。')

def generate_with_llm(text, *, endpoint, model, api_key_env='AGENT_LLM_API_KEY', timeout=60):
    """显式传入服务地址和模型；调用者应仅连接信任的模型服务。"""
    if not isinstance(text,str) or not text.strip():raise ValueError('EMPTY_REQUEST: 任务文本为空')
    require_explicit_boptest(text)
    if not endpoint.startswith(('http://','https://')):raise ValueError('INVALID_ENDPOINT')
    payload={'model':model,'temperature':0,'messages':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':text}],
             'response_format':{'type':'json_object'}}
    headers={'Content-Type':'application/json'}
    key=os.environ.get(api_key_env)
    if key:headers['Authorization']='Bearer '+key
    request=Request(endpoint,json.dumps(payload).encode(),headers,method='POST')
    with urlopen(request,timeout=timeout) as response:result=json.load(response)
    content=result['choices'][0]['message']['content']
    intent,config=parse_response(content)
    return dict(intent=intent,config=config,metadata=dict(model=model,response_model=result.get('model'),
        prompt_version=PROMPT_VERSION,temperature=0,usage=result.get('usage')))
