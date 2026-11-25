"""Week 5 一键流水线：核验冻结资产、评测、统计、绘图、回归与报告；保存本次清单。"""
from pathlib import Path
import sys
import os
import json
import hashlib
import subprocess
import platform
import importlib.metadata
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone, timedelta
from time import perf_counter
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]


class Tee:
    """阶段日志同时写入终端和本次运行目录，失败时也便于定位。"""
    def __init__(self,console,log): self.console,self.log=console,log
    def write(self,value):
        self.console.write(value); self.log.write(value)
        return len(value)
    def flush(self): self.console.flush(); self.log.flush()


def digest(path):
    """分块计算 SHA-256，避免原始预测较大时重复占用内存。"""
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def verify_outputs(config):
    """检查模型×horizon 覆盖、共同起点身份、数组形状及跨模式首步一致性。"""
    output=ROOT/config['output']; folder=output/'rollout'
    origins=pd.read_csv(folder/'origins.csv')
    expected=config['origins']['expected_test_episodes']*config['origins']['expected_origins_per_episode']
    if len(origins)!=expected or origins.duplicated(['episode_id','source_step']).any():
        raise ValueError('Origin identity/count mismatch')
    if set(origins.episode_id)!=set(config['data']['split_ids']['test']):
        raise ValueError('Origin episode IDs differ from protocol')
    summary=pd.read_csv(folder/'metrics_by_horizon.csv')
    single=pd.read_csv(output/'one_step/predictions.csv.gz')
    for spec in config['models']:
        rows=summary[summary.model==spec['name']]
        if set(rows.horizon)!=set(range(1,config['max_horizon']+1)):
            raise ValueError('Incomplete model/horizon coverage')
        if not rows.requested.eq(expected).all(): raise ValueError('Failure denominators changed')
        with np.load(folder/f'{spec["name"]}_predictions.npz') as arrays:
            shape=(expected,config['max_horizon'],2)
            if arrays['actual'].shape!=shape: raise ValueError('Invalid ground-truth axes')
            first=arrays['recorded_observation__free_running'][:,0]
            for condition in config['information']['conditions']:
                for mode in config['modes']:
                    key=condition+'__'+mode
                    if arrays[key].shape!=shape: raise ValueError('Invalid prediction axes')
                    np.testing.assert_allclose(arrays[key][:,0],first,atol=1e-5,equal_nan=True)
                    attempted=arrays[key+'__attempted']
                    if attempted.shape!=shape[:2] or (np.diff(attempted.astype(int),axis=1)>0).any():
                        raise ValueError('Invalid stopping/attempt mask')
            for index,variable in enumerate(('air','wall')):
                selected=single[(single.model==spec['name'])&(single.variable==variable)]
                paired=origins.merge(selected,on=['episode_id','source_step'],validate='one_to_one')
                if len(paired)!=expected: raise ValueError('Single-step origin mismatch')
                np.testing.assert_allclose(paired.predicted,first[:,index],atol=1e-5,equal_nan=True)
    families=summary.family.nunique()
    if families<3 or len(config['horizons'])<4:
        raise ValueError('Acceptance requires >=3 model families and >=4 horizons')
    return dict(status='passed',origins=expected,episodes=origins.episode_id.nunique(),
        model_configurations=len(config['models']),model_families=families,horizons=config['horizons'],
        full_horizon_count=config['max_horizon'],first_step_consistent=True)


def update_report(config,run_dir,manifest):
    """只更新带标记的自动验收段，保留人工分析、独立扩展与前面所有图表。"""
    output=ROOT/config['output']; report=output.parent/'week5.md'
    metrics=pd.read_csv(output/'rollout/metrics_across_seeds.csv')
    metrics=metrics[(metrics['mode']=='free_running')&(metrics.condition=='recorded_observation')&(metrics.definition=='terminal')]
    header='| 模型族 | 变量 | '+' | '.join(f'H={h}' for h in config['horizons'])+' |'
    divider='|---|---|'+'---:|'*len(config['horizons'])
    lines=['<!-- WEEK5_REPRODUCTION_START -->','', '## 一键复现：最近一次验收','',
        f'- 配置：`{manifest["config_path"]}`；SHA-256：`{manifest["config_sha256"]}`。',
        f'- 运行：`{manifest["run_id"]}`；设备 `{manifest["device"]}`；Python `{manifest["python"]}`。',
        f'- 验收：{manifest["verification"]["episodes"]} 个 episode、{manifest["verification"]["origins"]} 个共同起点；'
        f'{manifest["verification"]["model_families"]} 个模型族、{len(config["models"])} 个模型配置，覆盖 {config["horizons"]}。',
        '- 逐步日志与数组由 `rollout/origins.csv` 映射到 episode_id 和 source_step；数组第二轴是预测步长，第三轴是 air/wall。',
        '- 下面是给定天气观测的自由递推室温/墙温末端 RMSE（°C）；± 为训练种子 SD。其他条件、MAE 和前缀指标见完整表。','',header,divider]
    for (family,variable),group in metrics.groupby(['family','variable']):
        values=[]
        for horizon in config['horizons']:
            row=group[group.horizon==horizon].iloc[0]
            values.append(f'{row.rmse_c_mean:.5f} ± {row.rmse_c_seed_sd:.5f}' if np.isfinite(row.rmse_c_seed_sd) else f'{row.rmse_c_mean:.5f}')
        lines.append('| '+family+' | '+variable+' | '+' | '.join(values)+' |')
    relative=run_dir.relative_to(report.parent).as_posix()
    lines.extend(['',f'[本次运行清单]({relative}/run_manifest.json) · [验收结果]({relative}/verification.json) · [测试日志]({relative}/tests.log)',
        '', '[完整表格](figures/evaluation_tables.md) · [误差曲线](figures/terminal_rmse.png) · [有效比例](figures/valid_prediction_fraction.png) · [排序变化](figures/ranking_by_horizon.png)',
        '', '```bash','conda activate mvp','bash scripts/reproduce_week5.sh','```','',
        '此一键命令复现冻结的热工主实验，不重新训练或重新冻结数据。IB、投影对照和补充分组 bootstrap 沿用各自独立脚本，避免混入主协议。',
        '', '<!-- WEEK5_REPRODUCTION_END -->'])
    section='\n'.join(lines)+'\n'
    existing=report.read_text() if report.exists() else '# Week 5 评测报告\n'
    start,end='<!-- WEEK5_REPRODUCTION_START -->','<!-- WEEK5_REPRODUCTION_END -->'
    if start in existing and end in existing:
        left=existing.index(start); right=existing.index(end,left)+len(end)
        existing=existing[:left]+section+existing[right:]
    else: existing+='\n'+section
    report.write_text(existing,encoding='utf-8')
    return report


def reproduce(config_path,device='cpu'):
    """单配置完成主验收；失败也保存状态与错误，不输出虚假的完成标记。"""
    config_path=Path(config_path).resolve()
    config=yaml.safe_load(config_path.read_text())
    # 先验证冻结资产，再创建结果文件；版本不符不得自动重新冻结。
    for name,expected in config['sha256'].items():
        if digest(ROOT/name)!=expected: raise ValueError('Frozen asset mismatch: '+name)
    if config['environment']!='simple_thermal' or config['max_horizon']!=80:
        raise ValueError('One-command report currently supports the frozen 80-step thermal main protocol')
    now=datetime.now(timezone(timedelta(hours=8)))
    run_id=now.strftime('%Y%m%d_%H%M%S_%f')
    output=ROOT/config['output']; run_dir=output/'reproduction'/run_id
    run_dir.mkdir(parents=True,exist_ok=False)
    (run_dir/'config_snapshot.yaml').write_bytes(config_path.read_bytes())
    versions={}
    for name in ('torch','numpy','pandas','matplotlib','PyYAML','gymnasium'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='not installed'
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    status=subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True)
    sources={}
    for folder in ('evaluators','metrics','models','scripts','simulators','controllers','data','tests'):
        for path in sorted((ROOT/folder).iterdir()):
            if not path.is_file() or path.suffix not in ('.py','.sh'): continue
            relative=path.relative_to(ROOT).as_posix()
            sources[relative]=digest(path)
            target=run_dir/'code_snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
    manifest=dict(run_id=run_id,status='running',started_at=now.isoformat(),device=device,
        config_path=config_path.relative_to(ROOT).as_posix(),config_sha256=digest(config_path),
        python=platform.python_version(),python_executable=sys.executable,platform=platform.platform(),
        packages=versions,git_commit=git.stdout.strip() if git.returncode==0 else None,
        git_dirty=bool(status.stdout.strip()),git_status=status.stdout.splitlines(),
        code_sha256=sources,code_snapshot='code_snapshot/',input_sha256=config['sha256'],stages=[],artifact_sha256={})
    def save_manifest():
        (run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    def stage(name,operation):
        print(f'[Week5] {name}',flush=True); started=perf_counter()
        try:
            with (run_dir/'pipeline.log').open('a',encoding='utf-8') as log:
                log.write(f'\n[Week5] {name}\n')
                with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                    result=operation()
        except Exception as exc:
            manifest['stages'].append(dict(name=name,status='failed',seconds=perf_counter()-started,error=str(exc)))
            save_manifest()
            raise
        manifest['stages'].append(dict(name=name,status='passed',seconds=perf_counter()-started)); save_manifest()
        return result
    save_manifest(); begin=perf_counter()
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        torch.manual_seed(config['evaluation_seed']); np.random.seed(config['evaluation_seed'])
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        manifest['torch_threads']=torch.get_num_threads()
        manifest['torch_cuda_version']=torch.version.cuda
        from evaluators.prediction import evaluate_one_step
        from evaluators.rollout import evaluate_week5_rollouts
        stage('single-step',lambda:evaluate_one_step(config_path,ROOT,device))
        stage('rollout + episode/seed summary + paired differences',lambda:evaluate_week5_rollouts(config_path,ROOT,device))
        from scripts import plot_week5
        stage('tables and figures',lambda:plot_week5.main(config_path))
        verification=stage('traceability and acceptance',lambda:verify_outputs(config))
        (run_dir/'verification.json').write_text(json.dumps(verification,indent=2))
        manifest['verification']=verification
        def tests():
            names=['tests.test_summary','tests.test_week5_extensions','tests.test_validity',
                   'tests.test_week5_rollout','tests.test_week5_prediction','tests.test_metrics',
                   'tests.test_dynamics_api','tests.test_physics_constraints']
            result=subprocess.run([sys.executable,'-m','unittest',*names],cwd=ROOT,capture_output=True,text=True)
            (run_dir/'tests.log').write_text(result.stdout+result.stderr)
            if result.returncode: raise RuntimeError('Regression tests failed; see '+str(run_dir/'tests.log'))
        stage('regression checks',tests)
        report=stage('report',lambda:update_report(config,run_dir,manifest))
        (run_dir/'report_snapshot.md').write_bytes(report.read_bytes())
        artifacts=[]
        for folder in (output/'one_step',output/'rollout',output.parent/'figures'):
            artifacts.extend(path for path in folder.iterdir() if path.is_file() and path.suffix in ('.csv','.gz','.npz','.json','.md','.png','.pdf','.svg'))
        artifacts.extend([report,run_dir/'config_snapshot.yaml',run_dir/'verification.json',run_dir/'tests.log',run_dir/'pipeline.log',run_dir/'report_snapshot.md'])
        # 不将上次补充 bootstrap 或旧 QA 报告伪装成本次重新计算的产物。
        artifacts=[p for p in artifacts if p.name not in ('episode_bootstrap_ci.csv',) and '.collision.' not in p.name]
        manifest['artifact_sha256']={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(artifacts))}
        manifest['status']='complete'
        print(f'[Week5] COMPLETE: {report}\n[Week5] Manifest: {run_dir / "run_manifest.json"}',flush=True)
    except Exception as exc:
        manifest['status']='failed'; manifest['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest['seconds']=perf_counter()-begin
        manifest['finished_at']=datetime.now(timezone(timedelta(hours=8))).isoformat()
        save_manifest()
    return manifest
