"""总说明：按照冻结用例对 Agent 的配置、工具、错误识别及运行结果评分，不自行调用服务。"""
import json
from pathlib import Path

def score_cases(cases, outcomes):
    """outcomes 为逐条实际记录；缺失输出计失败，非适用指标不进入对应分母。"""
    ids=[c['id'] for c in cases]
    if len(ids)!=len(set(ids)):raise ValueError('测试集 ID 重复')
    lookup={o['id']:o for o in outcomes}
    if len(lookup)!=len(outcomes) or set(lookup)-set(ids):raise ValueError('输出 ID 重复或不属于测试集')
    results=[]
    for case in cases:
        out=lookup.get(case['id'],{})
        valid_case=case['expected_action']=='generate_and_validate'
        expected=case['expected_models']
        config_ok=bool(out.get('config_valid')) and out.get('models')==expected if valid_case else None
        tool_ok=(out.get('tool_choice')==('validate_config' if valid_case else 'reject_or_clarify')) if out.get('tool_choice_evaluated',True) else None
        error_ok=(out.get('tool_choice')=='reject_or_clarify' and not out.get('executed',False)) if not valid_case else None
        run_ok=(bool(out.get('executed')) and bool(out.get('audit_passed'))) if valid_case and out.get('execution_evaluated',False) else None
        results.append(dict(id=case['id'],category=case['category'],config_correct=config_ok,
                            tool_correct=tool_ok,error_identified=error_ok,run_success=run_ok,
                            missing=not bool(out),trace=out.get('trace'),failure=out.get('failure')))
    summary={}
    for key in ('config_correct','tool_correct','error_identified','run_success'):
        values=[r[key] for r in results if r[key] is not None]
        summary[key]=dict(numerator=sum(bool(v) for v in values),denominator=len(values),
                          rate=sum(bool(v) for v in values)/len(values) if values else None)
    return dict(summary=summary,cases=results)

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--cases',required=True);parser.add_argument('--outcomes',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cases=json.loads(Path(args.cases).read_text())['cases'];outcomes=json.loads(Path(args.outcomes).read_text())['cases']
    Path(args.output).write_text(json.dumps(score_cases(cases,outcomes),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
