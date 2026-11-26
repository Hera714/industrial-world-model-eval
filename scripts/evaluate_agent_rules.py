"""总说明：执行冻结测试集的规则基线，保存逐条记录与明确标注的离线评分。"""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agents.rule_baseline import run_rule
from agents.tools import invoke
from agents.evaluation_guard import may_execute
from evaluators.agent import score_cases

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',default='data/agent_eval/week8_cases.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true',help='对冻结用例中允许的任务执行真实 BOPTEST；服务故障即停止')
    args=parser.parse_args();path=ROOT/args.cases
    data=json.loads(path.read_text());outcomes=[]
    destination=ROOT/args.output;destination.parent.mkdir(parents=True,exist_ok=True)
    for case in data['cases']:
        # 先离线解析，再独立核对冻结答案，防止误解的指令触发真实运行。
        outcome=run_rule(case['request'],execute=False)
        eligible=may_execute(case,{'models':outcome['models']},outcome['config_valid'])
        if args.execute and eligible:
            outcome=run_rule(case['request'],execute=True,tool=invoke)
        outcomes.append(dict(id=case['id'],**outcome))
        destination.write_text(json.dumps(dict(status='running',outcomes=outcomes),ensure_ascii=False,indent=2))
        failure=next((e for e in outcome['trace'] if e['tool']=='run_benchmark' and not e['result']['ok']),None)
        if failure:
            destination.write_text(json.dumps(dict(status='stopped_service_failure',outcomes=outcomes,
                remaining=len(data['cases'])-len(outcomes),reason=failure['result']['error']),ensure_ascii=False,indent=2))
            raise RuntimeError('真实服务失败，已停止后续用例；记录见 '+str(destination))
    result=score_cases(data['cases'],outcomes)
    # 未执行真实仿真时，运行成功率并非已测得的 0%；明确置为未评估。
    if not args.execute:result['summary']['run_success']['reason']='离线模式未执行真实 BOPTEST'
    result['metadata']=dict(mode='real_boptest' if args.execute else 'offline_rule_baseline',case_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            execution_enabled=args.execute,
                            executed_real_simulator=any(row['executed'] for row in outcomes),used_llm=False)
    result['outcomes']=outcomes
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(destination)

if __name__=='__main__':main()
