"""总说明：评测本地模型自主选择第一个工具；只读取结果，绝不运行模拟器。"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.local_llm import LocalGenerator
from agents.react_llm import LocalToolDecider


def main():
    cases_file = ROOT / 'data/agent_eval/week8_cases.json'
    cases = json.loads(cases_file.read_text(encoding='utf-8'))['cases']
    manifest = json.loads((ROOT / 'configs/agent/local_model_manifest.json').read_text())
    model_path = ROOT / 'external/qwen-agent'
    with (model_path / manifest['weight']).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != manifest['weight_sha256']:
        raise ValueError('模型权重哈希不匹配')
    for name, expected in manifest['metadata_sha256'].items():
        if hashlib.sha256((model_path / name).read_bytes()).hexdigest() != expected:
            raise ValueError('模型元数据哈希不匹配：' + name)
    decider = LocalToolDecider(LocalGenerator(str(model_path), revision=manifest['revision']))
    destination = ROOT / 'reports/week8/tool_choice_first_action.json'
    rows = []
    for case in cases:
        started = time.perf_counter()
        try:
            action = decider(case['request'], [])
            chosen = action['tool']
            error = None
        except Exception as exc:
            action = None
            chosen = 'error'
            error = {'type': type(exc).__name__, 'message': str(exc)}
        # 无工具的 finish 仅表示拒绝或澄清；不执行任何工具或真实仿真。
        expected = 'validate_config' if case['expected_action'] == 'generate_and_validate' else 'finish'
        rows.append({'id': case['id'], 'expected_first_action': expected, 'chosen': chosen,
                     'correct': chosen == expected, 'action': action, 'error': error,
                     'elapsed_seconds': time.perf_counter() - started})
        destination.write_text(json.dumps({'status': 'running', 'cases': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    destination.write_text(json.dumps({'status': 'complete', 'scope': '仅模型自主选择首个工具；未调用工具或评测完整链路',
        'case_sha256': hashlib.sha256(cases_file.read_bytes()).hexdigest(), 'model_sha256': digest,
        'revision': manifest['revision'], 'correct': sum(row['correct'] for row in rows),
        'total': len(rows), 'cases': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
