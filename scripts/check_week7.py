"""总说明：离线核验配置、run 追溯和回归，并可选运行真实 BOPTEST 集成。"""
from pathlib import Path
import argparse, json, os, subprocess, sys
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.config import load_config
from benchmark.regression import check_run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--integration', action='store_true')
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute(): run = ROOT / run
    checks = []
    try:
        load_config(ROOT/'configs/benchmark/boptest_smoke.yaml', ROOT)
        checks.append({'name': '配置及输入哈希', 'status': 'passed'})
        policy = yaml.safe_load((ROOT/'configs/regression/boptest_smoke.yaml').read_text())
        errors = check_run(run, policy)
        checks.append({'name': 'run 回归', 'status': 'failed' if errors else 'passed', 'errors': errors})
        config = yaml.safe_load((run/'config.yaml').read_text())
        version = json.loads((run/'versions.json').read_text())
        hashes = json.loads((run/'input_hashes.json').read_text())
        status = json.loads((run/'status.json').read_text())
        trace = dict(run_id=status['run_id'], started_at=status['started_at'],
                     models=[dict(name=m['name'], sha256=m['checkpoint']['sha256'], seed=config['seed']['model'][m['name']]) for m in config['model']['instances']],
                     episodes=config['dataset']['evaluation_episodes'], seeds=config['seed'],
                     commit=version.get('commit'), git_dirty=version.get('git_dirty'), config_sha256=hashes['config_sha256'])
        if not trace['commit'] or not trace['episodes'] or not trace['config_sha256']: raise ValueError('追溯字段缺失')
        modules = ['tests.test_benchmark_boundaries','tests.test_benchmark_config','tests.test_benchmark_artifacts','tests.test_benchmark_loaders','tests.test_benchmark_runner','tests.test_week6_control','tests.test_metrics']
        result = subprocess.run([sys.executable,'-m','unittest',*modules], cwd=ROOT, capture_output=True, text=True)
        checks.append(dict(name='离线单元测试', status='passed' if result.returncode == 0 else 'failed', output=result.stderr[-2000:]))
        if args.integration:
            env=os.environ.copy();env['BOPTEST_INTEGRATION']='1'
            result=subprocess.run([sys.executable,'-m','unittest','tests.test_benchmark_integration'], cwd=ROOT, env=env, capture_output=True, text=True)
            checks.append(dict(name='真实服务集成',status='passed' if result.returncode == 0 else 'failed',output=(result.stdout+result.stderr)[-2000:]))
        else: checks.append(dict(name='真实服务集成',status='not_run'))
        passed=all(c['status'] != 'failed' for c in checks)
        print(json.dumps(dict(status='passed' if passed else 'failed',checks=checks,provenance=trace),ensure_ascii=False,indent=2))
        return 0 if passed else 1
    except Exception as error:
        print(json.dumps(dict(status='failed',error=str(error),checks=checks),ensure_ascii=False),file=sys.stderr)
        return 1

if __name__ == '__main__': raise SystemExit(main())
