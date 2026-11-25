"""总说明：Week7统一配置离线预检，仿真连接前校验必填项、单位、范围、时间支持与文件哈希。"""
from pathlib import Path
from urllib.parse import urlparse
import copy
import hashlib
import json
import math
import re
import yaml


class ConfigurationError(ValueError):
    """错误包含配置字段路径，方便在启动服务前定位问题。"""


def require(mapping, field, path):
    if not isinstance(mapping, dict) or field not in mapping:
        raise ConfigurationError(f'{path}.{field}: missing required field')
    return mapping[field]


def number(value, path, minimum=0, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f'{path}: expected a number')
    if not math.isfinite(value) or value < minimum or (integer and not isinstance(value, int)):
        raise ConfigurationError(f'{path}: expected finite value >= {minimum}' + (' (integer)' if integer else ''))
    return value


def choice(value, options, path):
    if value not in options:
        raise ConfigurationError(f'{path}: expected one of {options}, got {value!r}')


def checked_path(root, relative, path):
    if not isinstance(relative, str) or not relative:
        raise ConfigurationError(f'{path}: expected nonempty project-relative path')
    target = (root / relative).resolve()
    if Path(relative).is_absolute() or not target.is_relative_to(root):
        raise ConfigurationError(f'{path}: must stay inside project root')
    return target


def verify_file(root, spec, path, verify_files):
    file = checked_path(root, require(spec, 'path', path), path + '.path')
    expected = require(spec, 'sha256', path)
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ConfigurationError(f'{path}.sha256: expected lowercase SHA256')
    if verify_files:
        if not file.is_file():
            raise ConfigurationError(f'{path}.path: missing file {file}')
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ConfigurationError(f'{path}.sha256: file digest mismatch')
    return file


def validate_config(config, root, verify_files=True):
    """纯本地校验，不导入模拟器、不访问HTTP、不创建run目录。"""
    root = Path(root).resolve()
    config = copy.deepcopy(config)
    required = ['schema_version', 'model', 'env', 'dataset', 'seed', 'evaluator', 'horizon', 'metrics', 'output', 'failure_policy']
    for field in required:
        require(config, field, 'config')
    unknown = set(config) - set(required)
    if unknown:
        raise ConfigurationError(f'config: unknown fields {sorted(unknown)}')
    choice(config['schema_version'], ['1.0'], 'schema_version')
    model, env, data = config['model'], config['env'], config['dataset']
    for path, fields in [('model', ['instances','history_steps','device','observation_names','observation_units','exogenous_names','exogenous_units','input_features','output_features']),
        ('env',['type','url','version','upstream_commit','case','step_seconds','warmup_seconds','context_steps','episode_steps','area_m2','assets','action']),
        ('dataset',['mode','manifest','train_split','validation_split','evaluation_episodes','weather_information','context_policy','action_plan']),
        ('seed',['model','candidate','observation_fault']),
        ('evaluator',['tasks','prediction_modes','origin_stride_steps','fault_conditions','condition_matrix','faults','baseline']),
        ('horizon',['prediction_steps','mpc_steps','candidates','action_hold_steps']),
        ('metrics',['prediction','control','units','cost_weights','validity','aggregation']),
        ('output',['directory','run_name','formats','existing_run']),
        ('failure_policy',['invalid_prediction','planning_failure','advance_failure','retries','timeouts'])]:
        for field in fields:
            require(config[path],field,path)
        extra=set(config[path])-set(fields)
        if extra:raise ConfigurationError(f'{path}: unknown fields {sorted(extra)}')
    choice(env['type'],['boptest'],'env.type')
    parsed=urlparse(env['url'])
    if parsed.scheme not in ('http','https') or not parsed.hostname:
        raise ConfigurationError('env.url: expected absolute http(s) URL')
    if env['version']!='0.9.0' or env['case']!='bestest_hydronic_heat_pump':
        raise ConfigurationError('env: this contract currently supports frozen v0.9.0 bestest_hydronic_heat_pump')
    if not re.fullmatch('[0-9a-f]{40}',env['upstream_commit']):
        raise ConfigurationError('env.upstream_commit: expected full git commit')
    for field in ['step_seconds','warmup_seconds','context_steps','episode_steps']:
        number(env[field],'env.'+field,minimum=1,integer=True)
    if env['step_seconds']!=900:
        raise ConfigurationError('env.step_seconds: these checkpoints require 900 seconds')
    number(env['area_m2'],'env.area_m2',minimum=.001)
    if env['area_m2']!=192.:
        raise ConfigurationError('env.area_m2: case metadata requires192m²')
    number(model['history_steps'],'model.history_steps',minimum=2,integer=True)
    if model['history_steps']!=16 or env['context_steps']<model['history_steps']:
        raise ConfigurationError('model.history_steps/env.context_steps: require16-step history and sufficient context')
    if not re.fullmatch(r'cpu|cuda(?::[0-9]+)?',str(model['device'])):
        raise ConfigurationError('model.device: expected cpu or cuda[:index]')
    observation=['zone_c','supply_c','return_c','floor_heat_kw','electric_kw']
    if model['observation_names']!=observation or model['observation_units']!=['degC','degC','degC','kW','kW']:
        raise ConfigurationError('model.observation_names/units: incompatible frozen observation contract')
    if model['exogenous_names']!=['outdoor_c','solar_w_m2','sin_day','cos_day','sin_year','cos_year'] or model['exogenous_units']!=['degC','W/m2','1','1','1','1']:
        raise ConfigurationError('model.exogenous_names/units: incompatible weather/calendar contract')
    if model['input_features']!=12 or model['output_features']!=6:
        raise ConfigurationError('model: requires12 input features and6 outputs')
    action=env['action']
    for name in ['point','activation','unit','api_unit','bounds']:
        require(action,name,'env.action')
    if action['point']!='oveTSet_u' or action['activation']!='oveTSet_activate' or action['unit']!='degC' or action['api_unit']!='K':
        raise ConfigurationError('env.action: expected Celsius setpoint with Kelvin API and explicit activation')
    bounds=action['bounds']
    if not isinstance(bounds,list) or len(bounds)!=2:
        raise ConfigurationError('env.action.bounds: expected [lower,upper]')
    for value in bounds:number(value,'env.action.bounds',minimum=5)
    if not 5<=bounds[0]<bounds[1]<=35:
        raise ConfigurationError('env.action.bounds: outside API legal5..35°C range')
    if not isinstance(model['instances'],list) or not model['instances']:
        raise ConfigurationError('model.instances: expected nonempty list')
    names=[]
    for i,spec in enumerate(model['instances']):
        path=f'model.instances[{i}]'
        name=require(spec,'name',path);kind=require(spec,'kind',path)
        if not isinstance(name,str) or not name:raise ConfigurationError(path+'.name: expected name')
        choice(kind,['persistence','arx','mlp','gru','residual_mlp'],path+'.kind')
        verify_file(root,require(spec,'checkpoint',path),path+'.checkpoint',verify_files)
        names.append(name)
    if len(names)!=len(set(names)):raise ConfigurationError('model.instances: duplicate names')
    if not isinstance(config['seed']['model'],dict) or set(config['seed']['model'])!=set(names):
        raise ConfigurationError('seed.model: must map every selected model name')
    for seed in config['seed']['model'].values():
        if seed is not None:number(seed,'seed.model',integer=True)
    if verify_files:
        import torch
        for spec in model['instances']:
            try:
                checkpoint=torch.load(root/spec['checkpoint']['path'],map_location='cpu',weights_only=True)
            except Exception as error:
                raise ConfigurationError(f"model.instances.{spec['name']}: invalid checkpoint: {error}") from error
            from benchmark.loaders import check_checkpoint_dimensions
            check_checkpoint_dimensions(checkpoint, model['history_steps'], f"model.instances.{spec['name']}")
            if checkpoint.get('kind')!=spec['kind'] or checkpoint.get('length')!=model['history_steps']:
                raise ConfigurationError(f"model.instances.{spec['name']}: checkpoint kind/history mismatch")
            if spec['kind'] in ('mlp','gru','residual_mlp') and checkpoint.get('seed')!=config['seed']['model'][spec['name']]:
                raise ConfigurationError(f"seed.model.{spec['name']}: checkpoint seed mismatch")
    for field in ['candidate','observation_fault']:
        number(require(config['seed'][field],'base','seed.'+field),'seed.'+field+'.base',integer=True)
    choice(require(config['seed']['candidate'],'offset','seed.candidate'),['source_step'],'seed.candidate.offset')
    offsets=require(config['seed']['observation_fault'],'offset_by_evaluator','seed.observation_fault')
    if offsets!={'open_loop':'source_step','closed_loop':'start_day'}:
        raise ConfigurationError('seed.observation_fault.offset_by_evaluator: incompatible frozen random stream rule')
    asset=verify_file(root,env['assets'],'env.assets',verify_files)
    if verify_files:
        metadata=json.loads(asset.read_text())
        if metadata['version']!=env['version'] or metadata['commit']!=env['upstream_commit'] or metadata['case']!=env['case']:
            raise ConfigurationError('env.assets: version/commit/case mismatch')
        for relative,digest in metadata['sha256'].items():
            verify_file(root,dict(path=relative,sha256=digest),'env.assets.files',True)
    choice(data['mode'],['online','recorded'],'dataset.mode')
    choice(data['context_policy'],['replay_recorded_actions'],'dataset.context_policy')
    choice(data['action_plan'],['recorded_applied_actions'],'dataset.action_plan')
    choice(data['weather_information'],['persistence_at_origin','ideal_api_reference'],'dataset.weather_information')
    manifest=verify_file(root,data['manifest'],'dataset.manifest',verify_files)
    if not data['evaluation_episodes'] or len(data['evaluation_episodes'])!=len(set(data['evaluation_episodes'])):
        raise ConfigurationError('dataset.evaluation_episodes: nonempty unique identities required')
    if verify_files:
        ledger=json.loads(manifest.read_text());entries=ledger['episodes'];lookup={e['episode_id']:e for e in entries}
        if len(lookup)!=len(entries):raise ConfigurationError('dataset.manifest: duplicate episode')
        if not set(data['evaluation_episodes'])<=set(lookup):raise ConfigurationError('dataset.evaluation_episodes: unknown identity')
        for eid in data['evaluation_episodes']:
            entry=lookup[eid]
            if entry['split'] in (data['train_split'],data['validation_split']):
                raise ConfigurationError('dataset.evaluation_episodes: training/validation leakage')
            if entry['target_steps']<env['episode_steps'] or entry['context_steps']<env['context_steps']:
                raise ConfigurationError('dataset.manifest: episode exceeds recorded support')
        spans=sorted((e['warmup_start_s'],e['target_end_exclusive_s']) for e in entries)
        if any(a[1]>b[0] for a,b in zip(spans,spans[1:])):raise ConfigurationError('dataset.manifest: overlapping blocks')
        for entry in entries:verify_file(root,entry['file'],'dataset.manifest.episode',True)
    evaluator=config['evaluator']
    for field,options in [('tasks',['open_loop','closed_loop','robustness','policy_ranking']),('prediction_modes',['teacher_forcing','free_running']),('fault_conditions',['clean','noise_0p1','noise_0p3','missing_1','missing_4','goal_plus1'])]:
        values=evaluator[field]
        if not isinstance(values,list) or not values or len(values)!=len(set(values)):
            raise ConfigurationError('evaluator.'+field+': nonempty unique list required')
        for value in values:choice(value,options,'evaluator.'+field)
    number(evaluator['origin_stride_steps'],'evaluator.origin_stride_steps',minimum=1,integer=True)
    choice(evaluator['baseline'],['embedded','none'],'evaluator.baseline')
    if set(evaluator['condition_matrix'])!=set(evaluator['tasks']):
        raise ConfigurationError('evaluator.condition_matrix: one entry per enabled task required')
    for task,matrix in evaluator['condition_matrix'].items():
        episodes=require(matrix,'episodes','evaluator.condition_matrix.'+task)
        conditions=require(matrix,'conditions','evaluator.condition_matrix.'+task)
        if not episodes or not set(episodes)<=set(data['evaluation_episodes']):
            raise ConfigurationError('evaluator.condition_matrix.'+task+': unknown/empty episodes')
        if not conditions or not set(conditions)<=set(evaluator['fault_conditions']):
            raise ConfigurationError('evaluator.condition_matrix.'+task+': unknown/empty fault conditions')
    if 'policy_ranking' in evaluator['tasks'] and 'closed_loop' not in evaluator['tasks']:
        raise ConfigurationError('evaluator.tasks: policy_ranking requires generated closed_loop plans')
    expected_faults=set(evaluator['fault_conditions'])-{'clean'}
    if set(evaluator['faults'])!=expected_faults:
        raise ConfigurationError('evaluator.faults: definition required for every non-clean condition')
    for name,fault in evaluator['faults'].items():
        kind=require(fault,'kind','evaluator.faults.'+name)
        if kind=='temperature_noise':
            number(require(fault,'std_degC','evaluator.faults.'+name),'evaluator.faults.'+name+'.std_degC')
            if fault.get('channels')!=observation[:3]:raise ConfigurationError('evaluator.faults: temperature noise channels mismatch')
        elif kind=='sensor_missing':
            choice(require(fault,'channel','evaluator.faults.'+name),['supply_c'],'evaluator.faults.channel')
            number(require(fault,'burst_steps','evaluator.faults.'+name),'evaluator.faults.burst_steps',minimum=1,integer=True)
            number(require(fault,'repeat_period_steps','evaluator.faults.'+name),'evaluator.faults.repeat_period_steps',minimum=1,integer=True)
            if fault['burst_steps']>fault['repeat_period_steps']:raise ConfigurationError('evaluator.faults: burst exceeds repeat period')
            choice(require(fault,'fill','evaluator.faults.'+name),['last_valid'],'evaluator.faults.fill')
        elif kind=='comfort_shift':
            number(require(fault,'shift_degC','evaluator.faults.'+name),'evaluator.faults.shift_degC',minimum=-100)
        else:raise ConfigurationError('evaluator.faults: unsupported kind '+str(kind))
    horizon=config['horizon'];steps=horizon['prediction_steps']
    if not isinstance(steps,list) or not steps:
        raise ConfigurationError('horizon.prediction_steps: sorted unique nonempty steps required')
    for value in steps:number(value,'horizon.prediction_steps',minimum=1,integer=True)
    if steps!=sorted(set(steps)):raise ConfigurationError('horizon.prediction_steps: sorted unique steps required')
    if max(steps)>env['episode_steps']:raise ConfigurationError('horizon.prediction_steps: exceeds scored episode')
    for field,minimum in [('mpc_steps',1),('candidates',5),('action_hold_steps',1)]:
        number(horizon[field],'horizon.'+field,minimum=minimum,integer=True)
    if horizon['action_hold_steps']>horizon['mpc_steps']:raise ConfigurationError('horizon.action_hold_steps: exceeds planning horizon')
    metrics=config['metrics']
    if metrics['units']!={'temperature':'degC','power':'kW','energy':'kWh','time':'s','comfort_integral':'degC*h'}:
        raise ConfigurationError('metrics.units: incompatible physical units')
    for field,options in [('prediction',['mae','rmse','valid_fraction','physical_warning_fraction']),('control',['cost','energy_kwh','comfort_degree_hours','comfort_violation_rate','center_tracking_mae_c','decision_mean_ms','decision_p95_ms','fallback_count'])]:
        if not isinstance(metrics[field],list) or not metrics[field]:raise ConfigurationError('metrics.'+field+': nonempty list required')
        for value in metrics[field]:choice(value,options,'metrics.'+field)
    for field in ['comfort','energy','action_change']:
        number(require(metrics['cost_weights'],field,'metrics.cost_weights'),'metrics.cost_weights.'+field)
    validity=metrics['validity']
    for field in ['zone_degC','water_degC','interval_energy_kwh']:
        values=require(validity,field,'metrics.validity')
        if not isinstance(values,list) or len(values)!=2:raise ConfigurationError('metrics.validity.'+field+': expected two bounds')
        for value in values:number(value,'metrics.validity.'+field,minimum=-1000)
        if values[0]>=values[1]:raise ConfigurationError('metrics.validity.'+field+': bounds reversed')
    number(require(validity,'absolute_divergence','metrics.validity'),'metrics.validity.absolute_divergence',minimum=1)
    choice(require(validity,'nonfinite','metrics.validity'),['stop_keep_prefix'],'metrics.validity.nonfinite')
    choice(metrics['aggregation'],['episode_then_fixed_training_seeds'],'metrics.aggregation')
    output=config['output'];checked_path(root,output['directory'],'output.directory')
    if not isinstance(output['run_name'],str) or not re.fullmatch('[a-zA-Z0-9_-]+',output['run_name']):raise ConfigurationError('output.run_name: unsafe run name')
    if output['formats']!={'trajectory':'csv','prediction':'npz','metrics':'json','events':'jsonl','report':'md'}:
        raise ConfigurationError('output.formats: unsupported artifact formats')
    choice(output['existing_run'],['new_unique_run'],'output.existing_run')
    failure=config['failure_policy']
    for field,expected in [('invalid_prediction','stop_rollout_keep_prefix'),('planning_failure','release_to_embedded'),('advance_failure','stop_episode_no_retry')]:
        choice(failure[field],[expected],'failure_policy.'+field)
    for field in ['read_requests','advance_requests']:
        number(require(failure['retries'],field,'failure_policy.retries'),'failure_policy.retries.'+field,integer=True)
    if failure['retries']['advance_requests']!=0:raise ConfigurationError('failure_policy.retries.advance_requests: advancing cannot be retried')
    for field in ['http_seconds','planning_seconds']:
        number(require(failure['timeouts'],field,'failure_policy.timeouts'),'failure_policy.timeouts.'+field,minimum=.001)
    return config


def load_config(path, root=None, verify_files=True):
    """解析YAML并拒绝重复键；仅成功后交由未来Runner连接服务。"""
    class UniqueLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader,node,deep=False):
        result={}
        for key_node,value_node in node.value:
            key=loader.construct_object(key_node,deep=deep)
            if key in result:raise ConfigurationError(f'YAML duplicate key: {key}')
            result[key]=loader.construct_object(value_node,deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,construct_mapping)
    try:
        config=yaml.load(Path(path).read_text(),Loader=UniqueLoader)
    except yaml.YAMLError as error:
        raise ConfigurationError(f'YAML parse error: {error}') from error
    return validate_config(config,root or Path(__file__).resolve().parents[1],verify_files)
