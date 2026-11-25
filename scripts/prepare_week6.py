"""总说明：固定 BOPTEST v0.9.0、案例/FMUs/资源，归档本地构建补丁并冻结时间块。"""
from pathlib import Path
import hashlib
import json
import subprocess
import shutil
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    """只读取上游资产；记录构建输入摘要，不复制大型安装包进主仓库。"""
    upstream=ROOT/'external/project1-boptest'
    case=upstream/'testcases/bestest_hydronic_heat_pump/models'
    output=ROOT/'reports/week6/results/assets'; output.mkdir(parents=True,exist_ok=True)
    deployment=ROOT/'deployment/boptest'; deployment.mkdir(parents=True,exist_ok=True)
    patch=subprocess.check_output(['git','-C',str(upstream),'diff','--binary','--','service/provision/Dockerfile','service/web/Dockerfile','service/worker/Dockerfile'])
    (deployment/'local-build.patch').write_bytes(patch)
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    paths=[case/'wrapped.fmu',case/'config.json',case/'kpis.json',case/'days.json',upstream/'version.txt']
    paths+=list((case/'Resources').glob('*.csv'))
    inputs=[]
    for directory in ('service/web/apt-seed','service/worker/apt-seed','service/worker/installers'):
        for path in sorted((upstream/directory).glob('*')):
            if path.is_file() and not path.name.endswith('.part'):
                inputs.append(dict(path=str(path.relative_to(upstream)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest=dict(upstream='https://github.com/ibpsa/project1-boptest',commit=commit,version=(upstream/'version.txt').read_text().strip(),
        case='bestest_hydronic_heat_pump',area_m2=192.,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        build_patch_sha256=hashlib.sha256(patch).hexdigest(),local_build_inputs=inputs)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    config_path=ROOT/'configs/experiments/week6_boptest.yaml'
    if config_path.exists():
        old=yaml.safe_load(config_path.read_text())
        if old['upstream_commit']!=commit: raise ValueError('Existing frozen upstream differs')
        print('Assets saved; existing protocol preserved'); return
    # 先于采集固定时间块。每个块含7天预热、1天策略历史、2天评测/采集，块间不重叠。
    blocks={'train':[40,52,64,76,88,100,112,124], 'val':[136,148], 'id_test':[160,172],
        'ood_cold':[12,24], 'ood_warm':[196,208]}
    episodes=[]
    for split,days in blocks.items():
        for index,day in enumerate(days):
            episodes.append(dict(episode_id=f'{split}_day{day:03d}',split=split,start_day=day,
                seed=6042+day,policy='scheduled' if index%2==0 else 'random_hold'))
    config=dict(protocol_version='week6-boptest-v1',environment='boptest',url='http://127.0.0.1:80',
        upstream_commit=commit,case=manifest['case'],asset_manifest='reports/week6/results/assets/manifest.json',
        step_seconds=900,warmup_seconds=604800,history_steps=16,context_steps=96,episode_steps=192,
        action=dict(point='oveTSet_u',activation='oveTSet_activate',unit='K',experiment_bounds_c=[18.,24.],fallback='release_to_embedded'),
        measurements=['reaTZon_y','reaTSup_y','reaTRet_y','reaQFloHea_y','reaPHeaPum_y','reaPFan_y','reaPPumEmi_y'],
        state_names=['zone_c','supply_c','return_c','floor_heat_kw','electric_kw'],
        exogenous=['TDryBul','HGloHor'],horizons=[1,5,10,20],training_seeds=[42,43,44],
        episodes=episodes,split_rule='Each block owns [start-7days,start+3days); disjoint incl warmup/context.',
        ood_definition='Calendar/weather shifts reserved before training; quantify weather support after collection. Not guaranteed temperature extrapolation.',
        energy_label='Interval integral of reaPHeaPum_y + reaPFan_y + reaPPumEmi_y; native KPI delta checked. No setpoint-as-power.',
        control=dict(horizon=12,candidates=256,action_hold_steps=4,seeds=[42,43,44],time_budget_s=5.,
            comfort_weight=1.,energy_weight=1.,action_change_weight=.02),
        robustness=dict(noise_std_c=[0.,.1,.3],missing_burst_steps=[0,1,4],missing_sensor='supply_c',fill='last_valid',seed=64042),
        output='reports/week6/results')
    # 覆盖的时间支持包含预热，禁止分组间重叠。
    spans=sorted((e['start_day']-7,e['start_day']+3,e['episode_id']) for e in episodes)
    assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    config_path.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    (deployment/'README.md').write_text('''# BOPTEST 构建修改归档\n\n固定上游版本和 FMU 摘要见 reports/week6/results/assets/manifest.json。\n\n在相同上游 commit 的干净 checkout 中运行 `git apply deployment/boptest/local-build.patch`（补丁路径须相对于当前目录正确解析）。补丁包含本地 APT 镜像、非交互时区和本地 Miniconda 校验改动。\n\n补丁引用的 apt-seed 与 installers 文件未纳入 Git；manifest 列出路径、大小和 SHA256，重建前必须准备相同文件。现有 Docker 镜像并非该补丁本身，MinIO 自建镜像也需单独保留来源和镜像检查结果。禁止声称只靠此补丁即可离线重建全部镜像。\n''',encoding='utf-8')
    print(json.dumps(dict(version=manifest['version'],commit=commit,episodes=len(episodes),protocol=str(config_path))))


if __name__=='__main__': main()
