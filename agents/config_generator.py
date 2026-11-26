"""总说明：将已验证的结构化意图映射至冻结模型配置；LLM 输出必须先经过严格检查。"""
from pathlib import Path
import copy
import yaml
from agents.tools import validate_candidate

ROOT=Path(__file__).resolve().parents[1]


def config_from_intent(intent):
    """第一版只接受明确模型名称；禁止忽略额外字段和推测缺省任务。"""
    if not isinstance(intent,dict) or set(intent)!={'environment','models'}:
        raise ValueError('INVALID_INTENT: 需要且仅允许 environment 与 models')
    if intent['environment']!='boptest':
        raise ValueError('ENV_NOT_ALLOWED: 目前仅支持 boptest')
    names=intent['models']
    if not isinstance(names,list) or not names or any(not isinstance(n,str) for n in names) or len(set(names))!=len(names):
        raise ValueError('INVALID_MODELS: models 必须为非空且不重复的模型名称列表')
    config=yaml.safe_load((ROOT/'configs/benchmark/boptest_smoke.yaml').read_text())
    frozen=yaml.safe_load((ROOT/'configs/benchmark/boptest_week6.yaml').read_text())
    lookup={m['name']:m for m in frozen['model']['instances']}
    unknown=set(names)-set(lookup)
    if unknown: raise ValueError('MODEL_NOT_ALLOWED: '+str(sorted(unknown)))
    config['model']['instances']=[copy.deepcopy(lookup[n]) for n in names]
    config['seed']['model']={n:frozen['seed']['model'][n] for n in names}
    return validate_candidate(config)
