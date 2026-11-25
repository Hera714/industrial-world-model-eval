"""总说明：验收Week6冻结协议，逐文件重算SHA256并核查场景、模型、种子和完整比较矩阵。"""
from pathlib import Path
import json,hashlib
import yaml
ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'configs/experiments/week6_evaluation.yaml';cfg=yaml.safe_load(path.read_text())
    lock=json.loads((ROOT/cfg['lock_file']).read_text());failures=[]
    for relative,expected in lock['files'].items():
        file=ROOT/relative
        if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:failures.append(relative)
    if failures:raise ValueError('Frozen files missing/changed: '+str(failures))
    scenes=json.loads((ROOT/cfg['scenario_manifest']).read_text())['scenes']
    assert len(scenes)==cfg['expected']['data_blocks'] and len(cfg['models'])==8
    spans=sorted((s['warmup_start_s'],s['target_end_exclusive_s']) for s in scenes)
    assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    assert len({s['episode_id'] for s in scenes})==18
    names={m['name'] for m in cfg['models']};assert len(names)==8
    assert sum(len(s['control_conditions'])*9 for s in scenes)==153
    assert cfg['horizons']==[1,5,10,20] and cfg['seeds']['training']==[42,43,44]
    result=dict(status='passed',locked_files=len(lock['files']),scenes=18,checkpoints=8,expected_control_runs=153,
        entry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),lock_sha256=hashlib.sha256((ROOT/cfg['lock_file']).read_bytes()).hexdigest())
    out=ROOT/'reports/week6/results/protocol/verification.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
