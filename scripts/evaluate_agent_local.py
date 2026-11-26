"""总说明：对冻结用例逐条调用本地真实 LLM，记录输出与可选的真实 BOPTEST 评测。"""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agents.local_llm import LocalGenerator
from agents.tools import invoke
from evaluators.agent import score_cases
from agents.evaluation_guard import may_execute

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true',help='真实执行所有合法模型输出；可能耗时')
    args=parser.parse_args();directory=Path(args.model_dir);manifest=json.loads((ROOT/'configs/agent/local_model_manifest.json').read_text())
    for name,expected in manifest['metadata_sha256'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:raise ValueError('元数据哈希不符: '+name)
    with (directory/manifest['weight']).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=manifest['weight_sha256']:raise ValueError('模型哈希不符')
    generator=LocalGenerator(str(directory),revision=manifest['revision'])
    cases_path=ROOT/'data/agent_eval/week8_cases.json';cases=json.loads(cases_path.read_text())['cases'];outcomes=[]
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    for case in cases:
        start=time.perf_counter();row=dict(id=case['id'],tool_choice_evaluated=False,execution_evaluated=False,executed=False,audit_passed=False,trace=[],calls=1)
        try:
            response=generator(case['request'],endpoint=None,model=None)
            config=response['config'];row.update(models=response['intent']['models'],config_valid=True,tool_choice='validate_config',model_metadata=response['metadata'])
            checked=invoke('validate_config',{'config':config});row['trace'].append(dict(tool='validate_config',result=checked))
            row['config_valid']=bool(checked['ok'])
            # 测试集自身作为安全闸门：非法或错误理解的请求只评分，不向模拟器发送动作。
            eligible=may_execute(case,response['intent'],checked['ok'])
            row['execution_guard']=dict(eligible=eligible,reason='仅标准答案允许且模型选择精确一致时实跑')
            if args.execute and checked['ok'] and eligible:
                row['execution_evaluated']=True
                result=invoke('run_benchmark',{'config':config});row['trace'].append(dict(tool='run_benchmark',result=result))
                if result['ok']:
                    row['executed']=True
                    audit=invoke('summarize_results',{'run_directory':result['result']['run_directory']});row['trace'].append(dict(tool='summarize_results',result=audit))
                    row['audit_passed']=bool(audit['ok'] and audit['result']['passed'])
        except Exception as error:
            # 只有明确的不支持/歧义响应才是拒绝，通信、解析等异常保留为错误。
            reason=str(error)
            rejected=any(code in reason for code in ('AMBIGUOUS_REQUEST','ENV_NOT_ALLOWED','MODEL_NOT_ALLOWED','INVALID_INTENT','INVALID_MODELS','CONFIG_NOT_ALLOWED'))
            row.update(tool_choice='reject_or_clarify' if rejected else 'error',config_valid=False,models=[],failure=dict(type=type(error).__name__,message=reason,raw_response=getattr(error,'raw_response',None)))
        row['elapsed_seconds']=time.perf_counter()-start;outcomes.append(row)
        output.write_text(json.dumps(dict(status='running',cases=outcomes),ensure_ascii=False,indent=2))
        failed_run=next((e for e in row['trace'] if e['tool']=='run_benchmark' and not e['result']['ok']),None)
        if failed_run:
            output.write_text(json.dumps(dict(status='stopped_service_failure',cases=outcomes,remaining=len(cases)-len(outcomes),
                reason=failed_run['result']['error']),ensure_ascii=False,indent=2))
            raise RuntimeError('真实服务评测失败，已停止后续用例；记录见 '+str(output))
    score=score_cases(cases,outcomes)
    output.write_text(json.dumps(dict(status='completed',model_sha256=digest,revision=manifest['revision'],execution_enabled=args.execute,
        test_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),tool_selection_note='工具由固定编排调用，不评估 LLM 自主工具选择',summary=score['summary'],scores=score['cases'],cases=outcomes),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
