"""固定经验概率评分的分布声明、输入预测哈希与 horizon，不用测试结果选择参数。"""
from pathlib import Path
import hashlib
import yaml

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'configs/experiments/week5_probability_scores.yaml'
if target.exists(): raise FileExistsError('Probability scoring config already exists; do not overwrite it silently')
source=ROOT/'configs/experiments/week5_calibration.yaml'
calibration=yaml.safe_load(source.read_text())
folder=ROOT/calibration['output']
paths=[source,folder/'audit.json',folder/'test/origins.csv',folder/'test/failures.csv']
paths.extend(folder/'test'/f'{family}_members.npz' for family in calibration['families'])
config=dict(protocol_version='1.0-empirical-probability-scoring',
    source_calibration_config=str(source.relative_to(ROOT)),input_directory=calibration['output']+'/test',
    families=calibration['families'],member_seeds=[42,43,44],member_weights=[1/3]*3,
    distribution='equal-weight discrete empirical measure at the three original member predictions',
    score='exact positive-loss CRPS of empirical distribution, denominator M^2, lower is better',
    fair_iid_correction=False,calibration_transform='none: interval calibration does not define a full CDF',
    nll=dict(status='not_computed',reason='No continuous predictive density has been declared; the empirical measure is atomic.'),
    max_horizon=calibration['evaluation']['max_horizon'],horizons=calibration['evaluation']['horizons'],
    modes=calibration['evaluation']['modes'],conditions=list(calibration['evaluation']['information']['conditions']),
    variables=['air','wall'],unit='Celsius',
    invalid_members='any nonfinite member invalidates the fixed three-member score; retain denominator',
    aggregation='origins within episode, equal-weight episodes; ensemble is one predictor, no pseudo seed replicates',
    input_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    output='reports/week5/results/probability_scores')
target.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
print(target)
