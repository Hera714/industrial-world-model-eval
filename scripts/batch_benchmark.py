"""总说明：展开闭环评测矩阵，以独立进程和 run 目录执行，保存可恢复子任务账本。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, copy, hashlib, json, subprocess, sys, os, fcntl
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark.config import load_config
from benchmark.artifacts import atomic_write

def result_hashes(result):
    """保存结果目录所有文件摘要，恢复时校验原始轨迹及 episode 检查点。"""
    return {str(p.relative_to(result)):hashlib.sha256(p.read_bytes()).hexdigest() for p in result.rglob('*') if p.is_file()}

def verify_result(result, hashes):
    try:
        return bool(hashes) and json.loads((result/'status.json').read_text())['status']=='completed' and result_hashes(result)==hashes
    except (OSError,ValueError,KeyError): return False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.workers<=16: parser.error('workers 必须在 1..16')
    cfg=load_config(ROOT/args.config,ROOT)
    if set(cfg['evaluator']['tasks'])-{'closed_loop','robustness'}:
        parser.error('此入口仅支持闭环/鲁棒性矩阵，请先在配置中选择这两类任务')
    folder=(ROOT/args.output).resolve()
    if not folder.is_relative_to(ROOT): parser.error('output 必须位于项目内')
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'.coordinator.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: parser.error('此批次仍有协调器或子进程运行')
    # 配置校验已验证哈希；再把实际数据、资产与源码指纹固定在批次清单中。
    references=[cfg['dataset']['manifest'],cfg['env']['assets']]+[m['checkpoint'] for m in cfg['model']['instances']]
    ledger=json.loads((ROOT/cfg['dataset']['manifest']['path']).read_text())
    references += [e['file'] for e in ledger['episodes']]
    assets=json.loads((ROOT/cfg['env']['assets']['path']).read_text())
    references += [dict(path=p,sha256=h) for p,h in assets['sha256'].items()]
    inputs={r['path']:hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() for r in references}
    for directory in ('benchmark','evaluators','models','controllers','metrics','simulators'):
        for file in (ROOT/directory).glob('*.py'):
            inputs[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    inputs['run_benchmark.py']=hashlib.sha256((ROOT/'run_benchmark.py').read_bytes()).hexdigest()
    provenance=dict(configuration=cfg,inputs=inputs)
    manifest=folder/'batch_inputs.json'
    if manifest.exists() and json.loads(manifest.read_text())!=provenance:
        parser.error('配置、数据、模型或源码变化；请使用新的批次目录')
    atomic_write(manifest,json.dumps(provenance,sort_keys=True,indent=2))
    fingerprint=hashlib.sha256(yaml.safe_dump(cfg,sort_keys=True).encode()).hexdigest()
    jobs=[]
    for task,matrix in cfg['evaluator']['condition_matrix'].items():
        for episode in matrix['episodes']:
            for condition in matrix['conditions']:
                for model in cfg['model']['instances']:
                    identity=dict(task=task,episode=episode,condition=condition,model=model['name'])
                    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
                    jobs.append((key,identity,model))
    def execute(job):
        key,identity,model=job
        sub=folder/key;sub.mkdir(exist_ok=True)
        statepath=sub/'state.json'
        if statepath.exists():
            old=json.loads(statepath.read_text())
            if old['configuration_hash']!=fingerprint: raise ValueError('批量配置变化，必须使用新目录')
            if old['status']=='completed':
                result=Path(old['result'])
                if verify_result(result,old.get('result_hashes',{})): return old
                raise ValueError('已完成结果缺失或哈希不符：'+key)
        local=copy.deepcopy(cfg)
        local['model']['instances']=[model]
        local['seed']['model']={model['name']:cfg['seed']['model'][model['name']]}
        local['dataset']['evaluation_episodes']=[identity['episode']]
        local['evaluator']['tasks']=[identity['task']]
        local['evaluator']['condition_matrix']={identity['task']:dict(episodes=[identity['episode']],conditions=[identity['condition']])}
        local['evaluator']['baseline']='none'
        local['output']['directory']=str((sub/'runs').relative_to(ROOT))
        configpath=sub/'config.yaml';atomic_write(configpath,yaml.safe_dump(local,sort_keys=False))
        state=dict(identity=identity,configuration_hash=fingerprint,status='running')
        atomic_write(statepath,json.dumps(state))
        attempt=len(list(sub.glob('attempt-*.log')))+1
        with (sub/f'attempt-{attempt:03d}.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'run_benchmark.py'),'--config',str(configpath)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
            state.update(pid=process.pid,attempt=attempt)
            atomic_write(statepath,json.dumps(state,indent=2))
            process.wait()
        candidates=list((sub/'runs').glob('*/status.json'))
        result=max(candidates,key=lambda p:p.stat().st_mtime).parent if candidates else None
        state.update(status='completed' if process.returncode==0 else 'failed',returncode=process.returncode,result=str(result) if result else None)
        state['result_hashes']=result_hashes(result) if result and process.returncode==0 else {}
        if process.returncode==0 and not verify_result(result,state['result_hashes']):
            state['status']='failed';state['error']='子进程未产出完整结果'
        atomic_write(statepath,json.dumps(state,indent=2))
        return state
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=[]
        futures={pool.submit(execute,job):job for job in jobs}
        for future in as_completed(futures):
            try: results.append(future.result())
            except Exception as error:
                key,identity,_=futures[future]
                failure=dict(identity=identity,status='failed',error=repr(error))
                atomic_write(folder/key/'dispatch_failure.json',json.dumps(failure,indent=2))
                results.append(failure)
            atomic_write(folder/'summary.json',json.dumps(results,indent=2))
    atomic_write(folder/'summary.json',json.dumps(results,indent=2))
    return int(any(r['status']!='completed' for r in results))

if __name__=='__main__': raise SystemExit(main())
