"""Agent 工具：先校验参数和受控配置，再调用确定性评测与审核。"""
from pathlib import Path
import copy,hashlib,json,yaml
from benchmark.config import validate_config
from benchmark.runner import run_benchmark
from agents.schemas import validate_arguments
from agents.result_auditor import audit_run
from agents.errors import describe_error
ROOT=Path(__file__).resolve().parents[1]

def validate_candidate(config):
    policy=yaml.safe_load((ROOT/'configs/agent/policy.yaml').read_text())
    template=yaml.safe_load((ROOT/policy['template']).read_text())
    full=yaml.safe_load((ROOT/'configs/benchmark/boptest_week6.yaml').read_text())
    candidate=copy.deepcopy(config)
    # 仅允许从已冻结检查点选择模型；其余运行语义必须与短评测模板一致。
    allowed={m['name']:m for m in full['model']['instances']}
    models=candidate['model']['instances']
    if not models or len(models)>policy['max_models'] or any(m['name'] not in allowed or m!=allowed[m['name']] for m in models):raise ValueError('MODEL_NOT_ALLOWED: 检查点不在白名单')
    comparable=copy.deepcopy(candidate)
    comparable['model']['instances']=template['model']['instances']
    comparable['seed']['model']=template['seed']['model']
    if comparable!=template:raise ValueError('CONFIG_NOT_ALLOWED: 除模型选择外的配置字段不可更改')
    expected={m['name']:full['seed']['model'][m['name']] for m in models}
    if candidate['seed']['model']!=expected:raise ValueError('SEED_MISMATCH: 模型种子不匹配')
    return validate_config(candidate,ROOT)

def invoke(name,arguments,*,approved_config_hash=None):
    try:
        validate_arguments(name,arguments)
        if name=='list_envs':result={'environments':['boptest']}
        elif name=='validate_config':
            config=validate_candidate(arguments['config']);result={'valid':True,'models':[m['name'] for m in config['model']['instances']]}
        elif name=='run_benchmark':
            config=validate_candidate(arguments['config'])
            policy=yaml.safe_load((ROOT/'configs/agent/policy.yaml').read_text())
            digest=hashlib.sha256(json.dumps(arguments['config'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
            if len(config['model']['instances'])>policy['approval_required_above_models'] and approved_config_hash!=digest:
                raise ValueError('APPROVAL_REQUIRED: 请使用 LangGraph 人工审批')
            result={'run_directory':str(run_benchmark(config,ROOT))}
        else:
            path=Path(arguments['run_directory']).resolve();base=(ROOT/'reports/runs').resolve()
            if not path.is_relative_to(base):raise ValueError('PATH_NOT_ALLOWED: 结果目录不在白名单')
            result=audit_run(path)
        return {'ok':True,'result':result,'error':None}
    except Exception as error:
        return {'ok':False,'result':None,'error':describe_error(error)}
