"""总说明：只读复核已审批 LangGraph 的配置摘要、事件顺序和真实运行产物。"""
from pathlib import Path
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.graph import build_graph
from agents.result_auditor import audit_run


def main():
    pending = json.loads((ROOT / 'reports/week8/graph_approval_pending.json').read_text(encoding='utf-8'))
    expected_hash = pending['details']['config_sha256']
    with sqlite3.connect(ROOT / 'reports/week8/graph.sqlite', check_same_thread=False) as connection:
        graph = build_graph(checkpointer=SqliteSaver(connection))
        snapshot = graph.get_state({'configurable': {'thread_id': 'week8-approval-demo'}})
        state = snapshot.values
    events = state.get('events', [])
    names = [event.get('node') for event in events]
    run = Path(state.get('run_directory', ''))
    audit = audit_run(run)
    checks = {
        'terminal': not snapshot.next and state.get('status') == 'completed',
        'approved_config_hash': state.get('approved_hash') == expected_hash,
        'node_order': names == ['generate', 'validate', 'approval', 'execute', 'audit', 'summary'],
        'approval_recorded': any(e.get('node') == 'approval' and e.get('approved') is True
                                 and e.get('config_sha256') == expected_hash for e in events),
        'fresh_run_audit': audit['passed'],
        'graph_audit': bool(state.get('audit', {}).get('passed')),
    }
    result = {'passed': all(checks.values()), 'checks': checks, 'nodes': names,
              'approved_config_hash': expected_hash, 'run_directory': str(run),
              'failed_run_checks': [c for c in audit['checks'] if not c['passed']]}
    output = ROOT / 'reports/week8/graph_approved_verification.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
