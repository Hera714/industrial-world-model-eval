"""总说明：审核 Agent 会话与 benchmark 运行配置、步骤和最终审核之间的可追溯关系。"""
from pathlib import Path
import hashlib,json
import yaml
from agents.result_auditor import audit_run

ROOT=Path(__file__).resolve().parents[1]

def audit_session(session_directory):
    """只读审核；来源文件缺失或配置不一致明确失败。"""
    folder=Path(session_directory);checks=[]
    def check(name,value,detail):checks.append(dict(name=name,passed=bool(value),detail=str(detail)))
    try:
        state=json.loads((folder/'session.json').read_text())
        config=yaml.safe_load((folder/'config.yaml').read_text())
        events=state.get('events',[])
        check('session_status',state.get('status')=='completed',state.get('status'))
        names=[item.get('tool') for item in events]
        check('tool_order',names==['validate_config','run_benchmark','summarize_results'],names)
        check('tool_success',all(e.get('result',{}).get('ok') for e in events),names)
        check('model_metadata',bool(state.get('model_metadata',{}).get('model') and state.get('model_metadata',{}).get('revision')),state.get('model_metadata'))
        original=Path(next(e['result']['result']['run_directory'] for e in events if e.get('tool')=='run_benchmark'))
        relative=state.get('run_directory_relative')
        if relative is None and not original.exists() and original.parent.as_posix()=='/workspace/reports/runs':
            # 兼容本功能出现前、明确使用 /workspace 挂载的容器会话。
            relative='reports/runs/'+original.name
        if relative is not None:
            candidate=Path(relative)
            if candidate.is_absolute() or len(candidate.parts)!=3 or candidate.parts[:2]!=('reports','runs') or candidate.name in ('.','..'):
                raise ValueError('会话 run 相对路径非法')
            run=(ROOT/candidate).resolve()
            if not run.is_relative_to((ROOT/'reports/runs').resolve()):raise ValueError('会话 run 越界')
            check('portable_run_path',run.is_dir(),relative)
        else:
            run=original
        run_config=yaml.safe_load((run/'config.yaml').read_text())
        check('same_config',config==run_config,run/'config.yaml')
        check('run_report',(run/'report.md').is_file(),run/'report.md')
        result=audit_run(run)
        check('fresh_run_audit',result['passed'],[x['name'] for x in result['checks'] if not x['passed']])
        check('agent_audit',bool(state.get('audit',{}).get('passed')),folder/'session.json')
        check('request_nonempty',bool(state.get('request','').strip()),'request')
    except Exception as error:check('exception',False,repr(error))
    return dict(passed=all(c['passed'] for c in checks),checks=checks)
