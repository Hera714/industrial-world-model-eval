"""总说明：受控模型行动—工具观察循环；模型提出动作，工具白名单独立校验。"""
import time
import re
from pathlib import Path
import yaml
from agents.intent_guard import require_explicit_boptest
from agents.tools import invoke

ALLOWED={'list_envs','validate_config','run_benchmark','summarize_results'}

def run_react(request, decide, *, max_turns=8, tool=invoke):
    """decide 接收请求和历史观察并返回结构化 action；不记录或要求模型私有推理。"""
    if not isinstance(max_turns,int) or not 1<=max_turns<=32:raise ValueError('max_turns must be 1..32')
    history=[];start=time.monotonic()
    for index in range(max_turns):
        action = None
        try:
            action=decide(request,history.copy())
            if not isinstance(action,dict) or 'tool' not in action:raise ValueError('missing tool')
            name=action['tool']
            if name=='finish':
                # 完成必须由最近一次审核结果支持，避免直接伪造成功。
                audits=[x['observation'] for x in history if x['tool']=='summarize_results']
                verified=bool(audits and audits[-1].get('ok') and audits[-1].get('result',{}).get('passed'))
                return dict(status='completed' if verified else 'unverified',history=history,turns=index+1,
                            elapsed_seconds=time.monotonic()-start,summary=action.get('summary',''))
            if name not in ALLOWED:raise ValueError('UNKNOWN_TOOL: '+str(name))
            args=action.get('arguments',{})
            if name == 'run_benchmark':
                # 必须在本轮会话中校验完全相同的配置；禁止跳过校验或更换模型后执行。
                config = args.get('config')
                validated = any(x['tool'] == 'validate_config' and x['observation'].get('ok')
                                and x['arguments'].get('config') == config for x in history)
                if not validated:
                    raise ValueError('VALIDATION_REQUIRED: 运行前须校验同一配置')
                already_run = any(x['tool'] == 'run_benchmark' and x['observation'].get('ok')
                                  and x['arguments'].get('config') == config for x in history)
                if already_run:
                    raise ValueError('DUPLICATE_RUN: 相同配置已成功运行，请审核已有结果')
                require_explicit_boptest(request)
                names = [x['name'] for x in config['model']['instances']]
                policy=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/agent/policy.yaml').read_text())
                if len(names)>policy['approval_required_above_models']:
                    raise ValueError('APPROVAL_REQUIRED: 请使用 LangGraph 人工审批')
                text = request.lower()
                mentioned = re.findall(r'(?<![a-z0-9_])(residual_mlp_seed\d+|mlp_seed\d+|gru_seed\d+|persistence|arx)(?![a-z0-9_])', text)
                if names != mentioned:
                    raise ValueError('INTENT_MISMATCH: 仅允许运行用户明确指定的模型')
            if name == 'summarize_results':
                # 审核对象必须由当前会话的成功运行产生，不能借用旧运行冒充完成。
                target = args.get('run_directory')
                current_runs = [x['observation'].get('result', {}).get('run_directory')
                                for x in history if x['tool'] == 'run_benchmark'
                                and x['observation'].get('ok')]
                if not target or target not in current_runs:
                    raise ValueError('RUN_REQUIRED: 只能审核本次会话产生的运行结果')
            observation=tool(name,args)
        except Exception as error:
            name=action.get('tool','invalid') if isinstance(locals().get('action'),dict) else 'invalid'
            observation=dict(ok=False,error=dict(type=type(error).__name__,message=str(error)))
        history.append(dict(turn=index+1,tool=name,arguments=action.get('arguments',{}) if isinstance(action,dict) else {},raw_response=action.get('raw_response') if isinstance(action,dict) else None,observation=observation))
    return dict(status='turn_limit',history=history,turns=max_turns,elapsed_seconds=time.monotonic()-start)
