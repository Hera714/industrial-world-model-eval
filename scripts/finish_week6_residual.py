"""总说明：核验残差模型51个闭环与统一开环输出，合并比较表并生成架构/结果报告。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'reports/week6/results';out=base/'residual_mlp';cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text())
    ext=yaml.safe_load((ROOT/'configs/experiments/week6_boptest_extensions.yaml').read_text())
    models=json.loads((out/'models.json').read_text())['models'];records=[];missing=[]
    for e in cfg['episodes']:
        if e['split'] in ('train','val'):continue
        conditions=['clean']+(['noise_0p3','missing_4','goal_plus1'] if e['episode_id'] in ext['primary_reference_episodes'] else [])
        for c in conditions:
            for m in models:
                p=base/'closed_loop'/(e['episode_id']+'__'+m['name']+'__'+c+'.json')
                if not p.exists():missing.append(p.stem);continue
                r=json.loads(p.read_text());assert r['checkpoint_sha256']==m['sha256'] and r['status']=='complete'
                assert hashlib.sha256(p.with_suffix('.csv').read_bytes()).hexdigest()==r['csv_sha256']
                records.append(r)
    status=dict(status='passed' if not missing else 'in_progress',expected=51,completed=len(records),missing=missing)
    (out/'verification.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    if missing:return
    old=pd.read_csv(base/'summary/control_by_episode.csv');old=old[old.family.ne('residual_mlp')]
    new=pd.DataFrame(records);new['domain']=np.where(new.episode_id.isin(ext['primary_reference_episodes']),'reference',np.where(new.episode_id.eq('id_test_day172'),'calendar_shift',new.split))
    combined=pd.concat([old,new],ignore_index=True);combined.to_csv(out/'control_comparison.csv',index=False)
    oldopen=pd.read_csv(base/'open_loop/metrics_by_episode.csv');newopen=pd.read_csv(out/'open_loop/metrics_by_episode.csv')
    pd.concat([oldopen,newopen],ignore_index=True).to_csv(out/'open_comparison.csv',index=False)
    raw=pd.read_csv(out/'open_loop/endpoint_errors.csv.gz');assert len(raw)==304128
    keys=['episode_id','condition','mode','start','horizon','variable']
    baseline=pd.read_csv(base/'open_loop/endpoint_errors.csv.gz',usecols=keys+['actual'])
    truth=baseline.drop_duplicates(keys)
    paired=raw.merge(truth,on=keys,suffixes=('','_baseline'),validate='many_to_one')
    assert len(paired)==len(raw)
    np.testing.assert_allclose(paired.actual,paired.actual_baseline,atol=1e-10)
    config=dict(base_protocol='configs/experiments/week6_evaluation.yaml',model_manifest='reports/week6/results/residual_mlp/models.json',
        training_seeds=[42,43,44],additional_models=3,total_models=11,additional_controls=51,total_controls=204,
        architecture='192 -> 128 ReLU -> 64 ReLU -> 6',outputs='First5 physical observation increments plus direct interval energy',
        unchanged_comparison='same split/history/features/optimizer/early stopping/origins/horizons/MPC budget/cost/fallback',
        sha256={m['checkpoint']:m['sha256'] for m in models})
    (ROOT/'configs/experiments/week6_residual_evaluation.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    text='''# BOPTEST 历史残差 MLP 补充实验

输入16步×12特征，Flatten后192维，Linear(192,128)→ReLU→Linear(128,64)→ReLU→Linear(64,6)。前5维预测物理观测增量，反标准化后与当前观测相加；第6维直接预测区间能耗kWh。输出层没有强制非负，非法/负能耗沿用原有效性及候选过滤规则。

训练/验证数据、窗口、三个种子42/43/44、Adam学习率0.001、batch128、梯度限幅5、最大120epoch与早停耐心15均沿用直接历史MLP。增量输出尺度只在训练集拟合。残差目标与直接目标的标准化损失不可直接跨模型比较，比较应使用反标准化测试误差。

完成8日期块、352起点、两种评测模式、六种信息/故障条件、1/5/10/20步开环，共304128条端点记录。完成24个clean、18个故障、9个目标变化闭环，共51个。加上原153个，共204个控制回合。原八模型冻结清单保留，新检查点在本目录models.json中单独追溯。

完整比较表为open_comparison.csv与control_comparison.csv，原始残差预测在open_loop/raw_predictions.npz。闭环原始轨迹沿用../closed_loop，文件名含residual_mlp_seed。相同候选256条、horizon12、5秒检查预算、动作18～24°C和释放内置控制规则。不同预测目标带来的表现照实报告。
'''
    (ROOT/'reports/week6/residual_mlp.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
