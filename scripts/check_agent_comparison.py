"""重新审核真实规则/LLM 对照：配对配置、工具轨迹、run 哈希和未执行项。"""
from pathlib import Path
import hashlib
import json
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.result_auditor import audit_run


def main():
    cases_path = ROOT / 'data/agent_eval/week8_cases.json'
    cases = json.loads(cases_path.read_text(encoding='utf-8'))['cases']
    digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    inputs = {
        'rule': ROOT / 'reports/week8/rule_real_evaluation.json',
        'llm': ROOT / 'reports/week8/local_llm_real_evaluation.json',
    }
    loaded = {name: json.loads(path.read_text(encoding='utf-8')) for name, path in inputs.items()}
    checks = []

    def check(name, passed, detail):
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

    check('case_hash_rule', loaded['rule']['metadata']['case_sha256'] == digest, digest)
    check('case_hash_llm', loaded['llm']['test_sha256'] == digest, digest)
    lookups = {'rule': {r['id']: r for r in loaded['rule']['outcomes']},
               'llm': {r['id']: r for r in loaded['llm']['cases']}}
    check('coverage', all(set(items) == {c['id'] for c in cases} for items in lookups.values()), '20 条请求')
    check('unique_ids', len(lookups['rule']) == len(loaded['rule']['outcomes']) == len(cases)
          and len(lookups['llm']) == len(loaded['llm']['cases']) == len(cases), '两组请求均无重复身份')
    seen_runs = set()
    for case in cases:
        pair = []
        for name, lookup in lookups.items():
            row = lookup[case['id']]
            trace = row.get('trace', [])
            runs = [entry for entry in trace if entry['tool'] == 'run_benchmark']
            should_run = case['expected_action'] == 'generate_and_validate'
            if not should_run:
                check(f"{name}:{case['id']}:no_execution", not runs and not row.get('executed'), '拒绝或澄清')
                continue
            check(f"{name}:{case['id']}:tool_order", [e['tool'] for e in trace] ==
                  ['validate_config', 'run_benchmark', 'summarize_results'], '校验、运行、审核')
            if len(runs) != 1 or not runs[0]['result'].get('ok'):
                check(f"{name}:{case['id']}:run", False, '无唯一成功运行')
                continue
            original = Path(runs[0]['result']['result']['run_directory'])
            # 仅根据 run 目录名在项目内定位，跨挂载根目录时仍可验证。
            run = (ROOT / 'reports/runs' / original.name).resolve()
            if original.parent.name != 'runs' or not run.is_relative_to((ROOT / 'reports/runs').resolve()):
                check(f"{name}:{case['id']}:path", False, str(original))
                continue
            check(f"{name}:{case['id']}:unique", str(run) not in seen_runs, str(run.relative_to(ROOT)))
            seen_runs.add(str(run))
            audit = audit_run(run)
            check(f"{name}:{case['id']}:audit", audit['passed'],
                  [c['name'] for c in audit['checks'] if not c['passed']])
            check(f"{name}:{case['id']}:recorded_audit", bool(row.get('executed') and row.get('audit_passed')),
                  '记录中的运行及审核状态')
            try:
                pair.append(yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8')))
            except (OSError, yaml.YAMLError) as exc:
                check(f"{name}:{case['id']}:config_read", False, str(exc))
        if case['expected_action'] == 'generate_and_validate':
            check(f"pair:{case['id']}:config", len(pair) == 2 and pair[0] == pair[1], '配对配置相同')
    output = ROOT / 'reports/week8/comparison_reaudit.json'
    result = {'passed': all(c['passed'] for c in checks), 'dataset_sha256': digest,
              'runs_audited': len(seen_runs), 'checks': checks}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'runs_audited': len(seen_runs),
                      'failed': [c for c in checks if not c['passed']]}, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
