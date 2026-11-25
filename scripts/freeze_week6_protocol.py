"""总说明：为已执行的Week6协议生成统一验收入口、场景清单和输入锁文件，不改写实验结果。"""
from pathlib import Path
import json,hashlib
import yaml
ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    entry=ROOT/'configs/experiments/week6_evaluation.yaml'
    if entry.exists():raise FileExistsError('Frozen entry exists; verify it instead of overwriting')
    source=ROOT/'configs/experiments/week6_boptest.yaml';extra=ROOT/'configs/experiments/week6_boptest_extensions.yaml'
    cfg=yaml.safe_load(source.read_text());extension=yaml.safe_load(extra.read_text());base=ROOT/cfg['output']
    models=json.loads((base/'training/models.json').read_text())['models'];out=base/'protocol';out.mkdir(parents=True,exist_ok=True)
    scenes=[]
    for e in cfg['episodes']:
        day=e['start_day'];eid=e['episode_id'];split=e['split']
        domain='reference_support_matched' if eid in extension['primary_reference_episodes'] else ('calendar_shift' if eid=='id_test_day172' else ('seasonal_holdout' if split=='ood_cold' else ('warm_weather_holdout' if split=='ood_warm' else split)))
        conditions=['clean']+(['noise_0p3','missing_4','goal_plus1'] if eid in extension['primary_reference_episodes'] else []) if split not in ('train','val') else []
        scenes.append(dict(**e,analysis_domain=domain,warmup_start_s=(day-7)*86400,context_start_s=day*86400,
            target_start_s=(day+1)*86400,target_end_exclusive_s=(day+3)*86400,
            control_conditions=conditions,data_path=f'{cfg["output"]}/data/{eid}.csv'))
    scene_path=out/'scenario_manifest.json';scene_path.write_text(json.dumps(dict(time_intervals='half-open [start,end)',
        reference_semantics='Inside training outdoor range; does not prove joint-distribution equality',
        cold_semantics='Seasonal holdout, not temperature-support extrapolation',scenes=scenes),indent=2))
    inputs=[source,extra,ROOT/'docs/week6_protocol.md',base/'training/models.json',ROOT/cfg['asset_manifest'],scene_path]
    inputs+=sorted((base/'data').glob('*.csv'));inputs+=sorted((base/'data').glob('*.json'))
    inputs+=[ROOT/m['checkpoint'] for m in models]
    assets=json.loads((ROOT/cfg['asset_manifest']).read_text());inputs+=[ROOT/p for p in assets['sha256']]
    config=dict(protocol_version='week6-evaluation-frozen-1',scope='BOPTEST only; no IB/thermal scoring',
        protocol_document='docs/week6_protocol.md',execution_config=str(source.relative_to(ROOT)),extensions_config=str(extra.relative_to(ROOT)),
        scenario_manifest=str(scene_path.relative_to(ROOT)),lock_file=f'{cfg["output"]}/protocol/frozen_inputs.json',
        provenance='Consolidated acceptance index after execution; original scenario config and correction history are preserved, not claimed as a new preregistration.',
        models=[{k:v for k,v in m.items() if k in ('name','kind','seed','checkpoint','sha256')} for m in models],
        baseline='embedded controller, engineering reference with internal sensors',
        information_sets=dict(main='Current and historical public observations; future weather held at origin; known calendar/comfort schedule',
            ideal_reference='API perfect weather forecast, separately labelled',history_steps=16,history_token='observation before action, applied/current candidate action, weather/calendar',
            teacher_forcing='May observe next actual observation at each step',free_running='Own predicted observations only after origin'),
        horizons=cfg['horizons'],sampling=dict(origin_stride_steps=4,common_max_horizon=20,control_step_seconds=900,scored_episode_steps=192,context_steps=96,warmup_seconds=604800),
        seeds=dict(training=cfg['training_seeds'],observation_fault=64042,
            open_fault_rng='default_rng(64042 + origin source_step)',closed_fault_rng='default_rng(64042 + start_day)',
            candidate_rng='default_rng(604200 + episode step), identical for all models',
            simulator='Calendar/warmup deterministic; client reset seed does not create independent physical weather'),
        control_budget=cfg['control'],action=cfg['action'],robustness=cfg['robustness'],goal_shift=extension['ood_goal'],
        score=dict(formula='outside_comfort_deviation^2 * 0.25h + step_energy_kwh + 0.02*(action_C-previous_action_C)^2',
            comfort_bounds='Independent published schedule; goal_plus1 adds 1C to both bounds',energy='native ener_tot delta times area192m2',
            terminal_cost=0.,fallback='Release override to embedded; all costs retained',time_budget_check='Between batch predictions and at plan end; not preemptive'),
        validity=dict(nonfinite='Stop, retain prefix, later predictions invalid',divergence='Any absolute output above1000: stop',
            open_physical_warning='zone outside0..50C; water outside0..90C; negative power/energy; preserve original output',
            mpc_filter='finite; zone0..50C; interval energy0..10kWh; no valid candidates -> fallback'),
        aggregation=dict(order='window metrics within episode, then training-seed/date summaries as labelled',
            independent_unit='calendar block, not timestep or overlapping window',bootstrap_repeats=2000,
            comparison='Same date/history/budget across models; different-weather dates are distribution contrasts, not same-weather interventions',
            excluded_from_primary_reference=['id_test_day172'],primary_reference_episodes=extension['primary_reference_episodes']),
        expected=dict(data_blocks=18,checkpoints=8,clean_controls=72,fault_controls=54,goal_controls=27,total_controls=153))
    entry.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    inputs.append(entry)
    locked={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(inputs))}
    lock=out/'frozen_inputs.json';lock.write_text(json.dumps(dict(algorithm='sha256',files=locked),indent=2))
    print(json.dumps(dict(entry=str(entry.relative_to(ROOT)),locked_files=len(locked),scenes=len(scenes))))


if __name__=='__main__':main()
