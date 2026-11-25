"""总说明：检查 smoke 结果结构和宽松数值阈值；异常返回完整问题列表，不依赖逐位复现。"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import yaml


def check_run(path, policy):
    """读取已完成 run，不访问 BOPTEST；拒绝缺失、重复和非有限指标。"""
    path = Path(path)
    failures = []
    required = policy['required_files']
    missing = [name for name in required if not (path/name).is_file()]
    if missing:
        return ['missing files: ' + ', '.join(missing)]
    try:
        status = json.loads((path/'status.json').read_text())
        wrapper = json.loads((path/'metrics.json').read_text())
        metrics = wrapper['metrics']
        manifest = json.loads((path/'artifacts.json').read_text())
        for name in required:
            if name in ('status.json', 'artifacts.json'):
                continue
            if name not in manifest or hashlib.sha256((path/name).read_bytes()).hexdigest() != manifest[name]['sha256']:
                failures.append('artifact hash mismatch: '+name)
        if status['status'] != 'completed' or wrapper['status'] != 'available':
            failures.append('run not completed or metrics unavailable')
        seconds = (datetime.fromisoformat(status['finished_at']) - datetime.fromisoformat(status['started_at'])).total_seconds()
        if not 0 <= seconds <= policy['max_elapsed_seconds']:
            failures.append('run elapsed outside range')
        expected = {(model, mode, h) for model in policy['models'] for mode in policy['modes'] for h in policy['horizons']}
        rows = metrics['open_loop']
        identities = [(r['model'], r['mode'], r['horizon']) for r in rows]
        if len(identities) != len(expected) or set(identities) != expected:
            failures.append('open-loop matrix missing, duplicate or unexpected')
        if metrics['variables'] != policy['variables'] or metrics['units'] != policy['units']:
            failures.append('variable/unit mismatch')
        for row in rows:
            identity = str((row['model'], row['mode'], row['horizon']))
            if row['episode_id'] != policy['episode_id'] or row['condition'] != 'clean':
                failures.append('unexpected episode/condition '+identity)
            requested, valid, failed = row['requested'], row['valid'], row['failed']
            if requested != policy['requested_origins'] or valid+failed != requested or not 0 <= valid <= requested:
                failures.append('invalid prediction counts '+identity)
            if not isinstance(row['valid_fraction'], (int,float)) or not math.isfinite(row['valid_fraction']) or abs(row['valid_fraction']-valid/requested)>1e-9 or row['valid_fraction'] < policy['min_valid_fraction']:
                failures.append('invalid prediction fraction '+identity)
            for metric in ('mae', 'rmse'):
                values = row[metric]
                if len(values)!=6 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in values):
                    failures.append('invalid '+metric+' '+identity)
                elif values[0] > policy['max_zone_error_c']:
                    failures.append('zone error too large '+identity)
        controls = metrics['closed_loop']
        control_names = [r['model'] for r in controls]
        if len(control_names)!=len(policy['controllers']) or set(control_names)!=set(policy['controllers']):
            failures.append('closed-loop matrix missing, duplicate or unexpected')
        for row in controls:
            name=row['model']
            if row['task']!='closed_loop' or row['episode_id']!=policy['episode_id'] or row['condition']!='clean' or row['steps']!=policy['scored_steps']:
                failures.append('invalid closed-loop identity/steps '+name)
            for key, limit in policy['control_max'].items():
                value=row[key]
                if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=limit:
                    failures.append('invalid '+key+' '+name)
        with np.load(path/'predictions.npz', allow_pickle=False) as arrays:
            for model in policy['models']:
                for mode in policy['modes']:
                    prefix=f"open__{policy['episode_id']}__{model}__clean__{mode}"
                    for suffix in ('prediction','actual','valid','origins'):
                        if prefix+'__'+suffix not in arrays:
                            failures.append('missing prediction array '+prefix+'__'+suffix)
    except (KeyError, TypeError, ValueError, OSError, ZeroDivisionError) as error:
        failures.append(f'invalid result structure: {type(error).__name__}: {error}')
    return failures


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--policy',default='configs/regression/boptest_smoke.yaml')
    args=parser.parse_args()
    policy=yaml.safe_load(Path(args.policy).read_text())
    failures=check_run(args.run,policy)
    print(json.dumps(dict(status='failed' if failures else 'passed', failures=failures),ensure_ascii=False,indent=2))
    return int(bool(failures))


if __name__ == '__main__': raise SystemExit(main())
