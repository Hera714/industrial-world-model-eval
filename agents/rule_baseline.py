"""总说明：明确受限的规则基线，输出与 Agent 评测器一致的逐用例记录。"""
import re,time
from agents.config_generator import config_from_intent

def run_rule(request, *, execute=False, tool=None):
    """仅解析明确包含 BOPTEST 和检查点全名的短请求；其他请求拒绝，不猜测。"""
    start=time.perf_counter()
    text=request.lower()
    candidates=re.findall(r'(?<![a-z0-9_])(residual_mlp_seed\d+|mlp_seed\d+|gru_seed\d+|persistence|arx)(?![a-z0-9_])',text)
    allowed=re.fullmatch(r'\s*(?:运行\s*boptest\s*[，,]\s*比较\s*persistence\s*和\s*arx|用\s*boptest\s*测试\s*(?:arx|persistence)|boptest\s*上测试\s*(?:residual_mlp_seed\d+|mlp_seed\d+|gru_seed\d+))\s*',text)
    result=dict(tool_choice='reject_or_clarify',config_valid=False,models=[],executed=False,audit_passed=False,
                calls=0,elapsed_seconds=0.,cost=None,trace=[],execution_evaluated=execute)
    if allowed:
        try:
            config=config_from_intent(dict(environment='boptest',models=candidates))
            result.update(tool_choice='validate_config',config_valid=True,models=candidates)
            if execute:
                if tool is None:raise ValueError('实际执行必须提供工具')
                checked=tool('validate_config',{'config':config});result['trace'].append(dict(tool='validate_config',result=checked))
                if checked['ok']:
                    execution=tool('run_benchmark',{'config':config});result['trace'].append(dict(tool='run_benchmark',result=execution))
                    if execution['ok']:
                        result['executed']=True
                        audit=tool('summarize_results',{'run_directory':execution['result']['run_directory']})
                        result['trace'].append(dict(tool='summarize_results',result=audit));result['audit_passed']=bool(audit['ok'] and audit['result']['passed'])
        except Exception as error:
            result.update(tool_choice='reject_or_clarify',failure=str(error))
    result['calls']=len(result['trace']);result['elapsed_seconds']=time.perf_counter()-start
    return result
