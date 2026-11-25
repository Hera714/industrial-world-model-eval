"""总说明：依据天气元数据修正 ID 口径并冻结目标变化扩展；不根据模型测试误差挑日期。"""
from pathlib import Path
import yaml,json
ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'configs/experiments/week6_boptest.yaml';cfg=yaml.safe_load(path.read_text())
    extension=ROOT/'configs/experiments/week6_boptest_extensions.yaml'
    if not extension.exists():
        extension.write_text(yaml.safe_dump(dict(protocol_version='week6-extension-v1',
            reason='Day172 has 18.23% outside training outdoor range; preserve original result and label calendar shift, not pure ID.',
            selection_basis='Weather CSV only, no model-error/date optimization; day292/day340 have min/max inside training outdoor support.',
            primary_reference_episodes=['id_test_day160','id_test_day292','id_test_day340'],
            supplemental_calendar_episode='id_test_day172',
            ood_goal=dict(condition='goal_plus1',comfort_bound_shift_c=1.,physical_weather='identical to reference',model_retrained=False),
            weather_definition='cold blocks are seasonal holdouts, not range extrapolation; report support fractions for warm blocks',
            closed_loop_robustness=dict(episodes=['id_test_day160','id_test_day292','id_test_day340'],
                models='all frozen models plus embedded',conditions=['noise_0p3','missing_4']),
            seed_semantics='3 training seeds, deterministic physical date blocks; planning seed fixed, not independent physical replication'),sort_keys=False))
    for day in (292,340):
        eid=f'id_test_day{day:03d}'
        if not any(e['episode_id']==eid for e in cfg['episodes']):
            cfg['episodes'].append(dict(episode_id=eid,split='id_test',start_day=day,seed=6042+day,policy='scheduled' if day==292 else 'random_hold'))
    cfg['control']['planning_seed']=42
    spans=sorted((e['start_day']-7,e['start_day']+3) for e in cfg['episodes'])
    assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    print('Frozen supplemental ID and goal-shift protocol; original holdouts preserved')


if __name__=='__main__':main()
