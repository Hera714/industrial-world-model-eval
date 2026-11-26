"""确定性审核 run 产物、哈希、单位及指标数值；不会修改指标。"""
from pathlib import Path
import hashlib,json,math,csv
import yaml

def audit_run(directory):
    path=Path(directory);checks=[]
    def check(name,passed,detail):checks.append(dict(name=name,passed=bool(passed),detail=str(detail)))
    try:
        required=['config.yaml','status.json','metrics.json','trajectories.csv','input_hashes.json','artifacts.json']
        for name in required:check('file:'+name,(path/name).is_file(),path/name)
        if not all(x['passed'] for x in checks):return dict(passed=False,checks=checks)
        status=json.loads((path/'status.json').read_text());metrics=json.loads((path/'metrics.json').read_text());manifest=json.loads((path/'artifacts.json').read_text());config=yaml.safe_load((path/'config.yaml').read_text())
        check('completion',status.get('status')=='completed' and metrics.get('status')=='available',status.get('status'))
        check('run_id',metrics.get('metrics',{}).get('run_id')==status.get('run_id'),status.get('run_id'))
        for name,entry in manifest.items():
            file=(path/name).resolve()
            ok=file.is_relative_to(path.resolve()) and file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest()==entry.get('sha256')
            check('hash:'+name,ok,file)
        required_manifest={'config.yaml','versions.json','input_hashes.json','metrics.json','trajectories.csv','events.jsonl','run.log'}
        check('manifest_required',required_manifest <= set(manifest),sorted(required_manifest-set(manifest)))
        hashes=json.loads((path/'input_hashes.json').read_text())
        snapshot=(path/'config.yaml').read_bytes()
        check('config_hash',hashlib.sha256(snapshot).hexdigest()==hashes.get('config_sha256'),'config.yaml')
        root=Path(__file__).resolve().parents[1]
        for relative,digest in hashes.get('files',{}).items():
            file=(root/relative).resolve()
            ok=file.is_relative_to(root) and file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest()==digest
            check('input_hash:'+relative,ok,file)
        check('input_manifest_nonempty',bool(hashes.get('files')),'input_hashes.json')
        with (path/'trajectories.csv').open(newline='',encoding='utf-8') as stream:
            actual_rows=sum(1 for _ in csv.DictReader(stream))
        check('trajectory_count',actual_rows==status.get('trajectory_rows'),f"{actual_rows}/{status.get('trajectory_rows')}")
        data=metrics.get('metrics') or {}
        expected_models={m['name'] for m in config['model']['instances']}
        tasks=config['evaluator']['tasks']
        if 'open_loop' in tasks:
            expected={(m,e,c,mode,h) for m in expected_models for e in config['evaluator']['condition_matrix']['open_loop']['episodes'] for c in config['evaluator']['condition_matrix']['open_loop']['conditions'] for mode in config['evaluator']['prediction_modes'] for h in config['horizon']['prediction_steps']}
            actual=[(r['model'],r['episode_id'],r['condition'],r['mode'],r['horizon']) for r in data.get('open_loop',[])]
            check('open_loop_matrix',len(actual)==len(expected) and set(actual)==expected,f'{len(actual)}/{len(expected)}')
        for task in ('closed_loop','robustness'):
            if task in tasks:
                models=expected_models | ({'embedded'} if config['evaluator']['baseline']=='embedded' else set())
                expected={(m,e,c) for m in models for e in config['evaluator']['condition_matrix'][task]['episodes'] for c in config['evaluator']['condition_matrix'][task]['conditions']}
                actual=[(r['model'],r['episode_id'],r['condition']) for r in data.get('closed_loop',[]) if r.get('task')==task]
                check(task+'_matrix',len(actual)==len(expected) and set(actual)==expected,f'{len(actual)}/{len(expected)}')
        check('units',config['metrics']['units']=={'temperature':'degC','power':'kW','energy':'kWh','time':'s','comfort_integral':'degC*h'},config['metrics']['units'])
        check('variable_units',data.get('variables')==['zone_c','supply_c','return_c','floor_heat_kw','electric_kw','interval_energy_kwh'] and data.get('units')==['degC','degC','degC','kW','kW','kWh'],data.get('units'))
        for row in data.get('closed_loop',[]):
            initial=row.get('native_context_kpi') or {};final=row.get('native_final_kpi') or {}
            try:
                native_delta=(float(final['ener_tot'])-float(initial['ener_tot']))*float(config['env']['area_m2'])
                actual=float(row['energy_kwh'])
                consistent=math.isfinite(native_delta) and math.isfinite(actual) and math.isclose(native_delta,actual,rel_tol=1e-7,abs_tol=1e-7)
            except (KeyError,TypeError,ValueError):consistent=False
            check('native_energy_conversion:'+str(row.get('model')),consistent,'单位面积累计差 × 建筑面积 = 评分总 kWh')
        rows=metrics.get('metrics',{}).get('open_loop',[])+metrics.get('metrics',{}).get('closed_loop',[])
        def finite(value):
            if isinstance(value,dict):return 'nonfinite' not in value and all(finite(v) for v in value.values())
            if isinstance(value,list):return all(finite(v) for v in value)
            if isinstance(value,float):return math.isfinite(value)
            return True
        check('metrics_finite',all(finite(row) for row in rows),'逐条指标')
        for row in data.get('open_loop',[]):
            label=f"{row.get('model')}:{row.get('episode_id')}:{row.get('horizon')}"
            requested=row.get('requested');valid=row.get('valid');failed=row.get('failed');fraction=row.get('valid_fraction')
            consistent=(isinstance(requested,int) and requested>0 and isinstance(valid,int) and isinstance(failed,int)
                        and 0<=valid<=requested and failed==requested-valid and isinstance(fraction,(int,float))
                        and math.isfinite(fraction) and abs(fraction-valid/requested)<1e-9)
            check('prediction_counts:'+label,consistent,f'{valid}/{requested}; failed={failed}')
            check('prediction_all_valid:'+label,consistent and failed==0,'失败预测必须明确报告，不视为完全通过')
        for row in data.get('closed_loop',[]):
            label=f"{row.get('model')}:{row.get('episode_id')}"
            check('control_steps:'+label,row.get('steps')==config['env']['episode_steps'],row.get('steps'))
            fallback=row.get('fallback_count')
            check('fallback_count:'+label,isinstance(fallback,int) and 0<=fallback<=row.get('steps',-1),fallback)
        check('trajectory_nonempty',status.get('trajectory_rows',0)>0 and (path/'trajectories.csv').stat().st_size>0,status.get('trajectory_rows'))
    except Exception as error:check('audit_exception',False,repr(error))
    return dict(passed=all(x['passed'] for x in checks),checks=checks)
