"""首次创建独立概率校准配置；不会更改主评测切分、模型或既有协议。"""
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'configs/experiments/week5_calibration.yaml'
if target.exists(): raise FileExistsError('Calibration config already exists; do not silently overwrite it')
protocol=yaml.safe_load((ROOT/'configs/experiments/week5_evaluation.yaml').read_text())
config=dict(protocol_version='1.0-ensemble-interval-calibration',evaluation=protocol,
    families=['mlp','residual'],ensemble_seeds=[42,43,44],device='cpu',nominal_coverage=.9,
    scale_floor_c=.01,uncalibrated_z=1.6448536269514722,
    method='absolute ensemble error / max(sample ensemble SD,0.01C), per horizon and variable',
    calibration='validation pooled origins; all episodes have 65 origins; not an independent conformal calibration split',
    quantile='ceil((n+1)*nominal)-th smallest finite score; insufficient scores produce unavailable intervals',
    test='apply frozen q; no target-dependent fitting or model/threshold selection',
    interpretation='pointwise intervals; dependent origins and reused validation prevent a strict coverage guarantee',
    output='reports/week5/results/calibration')
target.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
print(target)
