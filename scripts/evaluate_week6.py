"""总说明：一个配置编排Week6十一模型训练/评测/审核，保留每次运行状态与失败日志。"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,uuid,hashlib
import yaml
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/experiments/week6_pipeline.yaml')
    parser.add_argument('--stage',choices=['verify','report','reproduce'],default='verify');args=parser.parse_args()
    config=yaml.safe_load((ROOT/args.config).read_text())
    folder=ROOT/config['run_directory']/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
    state=dict(status='running',config=args.config,stage=args.stage,steps=[])
    snapshot=yaml.safe_dump(config,sort_keys=False)
    (folder/'config_snapshot.yaml').write_text(snapshot)
    state['configuration_sha256']=hashlib.sha256(snapshot.encode()).hexdigest()
    status=folder/'status.json'
    env=os.environ.copy();env.pop('WEEK6_MODELS_MANIFEST',None);env.pop('WEEK6_OPEN_OUTPUT',None)
    groups=['verify'] if args.stage=='verify' else (['report','verify'] if args.stage=='report' else ['reproduce','report','verify'])
    try:
        for group in groups:
            for step in config[group]:
                entry=dict(command=step['command'],status='running');state['steps'].append(entry);status.write_text(json.dumps(state,indent=2))
                process_env=env|step.get('env',{})
                print('Running '+' '.join(step['command']),flush=True)
                with (folder/f'{len(state["steps"]):02d}.log').open('w') as log:
                    result=subprocess.run([sys.executable]+step['command'],cwd=ROOT,env=process_env,stdout=log,stderr=subprocess.STDOUT)
                entry['returncode']=result.returncode;entry['status']='passed' if result.returncode==0 else 'failed'
                if result.returncode:raise RuntimeError('Step failed; see '+str(folder))
        state['status']='passed'
    except Exception as error:
        state['status']='failed';state['error']=repr(error);raise
    finally:status.write_text(json.dumps(state,indent=2))
    print('Week6 '+args.stage+' passed; '+str(status),flush=True)


if __name__=='__main__':main()
