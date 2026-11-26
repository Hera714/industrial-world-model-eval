"""总说明：按用户、环境、标准答案和评测四个维度审核 Agent 测试用例。"""
import hashlib,json
from pathlib import Path

def audit_cases(path, model_names):
    """规则审核可发现结构和白名单问题，语义正确性仍需人工核验。"""
    path=Path(path);data=json.loads(path.read_text());cases=data['cases'];issues=[]
    ids=set()
    for case in cases:
        cid=case.get('id')
        if not cid or cid in ids:issues.append(dict(case=cid,component='User',problem='缺少或重复 ID'))
        ids.add(cid)
        if not str(case.get('request','')).strip():issues.append(dict(case=cid,component='User',problem='请求为空'))
        valid=case.get('expected_action')=='generate_and_validate'
        if valid and not set(case.get('expected_models',[]))<=set(model_names):issues.append(dict(case=cid,component='Environment',problem='模型不在冻结清单'))
        if valid and not case.get('expected_models'):issues.append(dict(case=cid,component='Ground Truth',problem='合法任务缺少目标模型'))
        rubric=case.get('rubric',{})
        if not isinstance(rubric,dict) or set(rubric)!={'intent','configuration','tool','result'}:issues.append(dict(case=cid,component='Evaluation',problem='缺少可检查评分维度'))
        if valid and case.get('expected_tools')!=['validate_config','run_benchmark','summarize_results']:issues.append(dict(case=cid,component='Ground Truth',problem='合法任务工具顺序不完整'))
        if not valid and case.get('expected_tools')!=[]:issues.append(dict(case=cid,component='Ground Truth',problem='拒绝任务不应自动执行'))
    if len(cases)!=20:issues.append(dict(case=None,component='Evaluation',problem='用例数不是20'))
    return dict(status='passed' if not issues else 'failed',cases=len(cases),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),issues=issues,
                limitations=['规则未验证自然语言语义、标准答案正确性或 LLM 评分可靠性，仍需人工复核。'])
