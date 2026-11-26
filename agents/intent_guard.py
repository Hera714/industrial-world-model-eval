"""执行前对用户原文进行保守检查；未知语义仍需模型和用户共同确认。"""
import re
def require_explicit_boptest(request):
    text=request.lower()
    if 'boptest' not in text or re.search(r'(?<![a-z])ib(?![a-z])|industrial benchmark|industrialbenchmark|sinergym|citylearn|energyplus',text):
        raise ValueError('ENV_NOT_ALLOWED: 任务须明确使用 BOPTEST，且不能混用其他环境')
