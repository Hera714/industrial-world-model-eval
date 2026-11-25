"""评估三个动力学基线的单步与多步误差，保存原始预测和物理单位指标。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.dynamics_base import PersistenceDynamics
from models.system_identification import LinearDynamics
from models.action_mlp import ActionMLP
from evaluators.rollout import evaluate_rollouts
from metrics.prediction import mae, rmse


def main():
    """只用训练集拟合线性基线，使用固定 MLP 检查点评估隔离测试集。"""
    folder = ROOT/'data/processed/week4_thermal'
    # 兼容编辑器为 CSV 表头和单元格补齐的空白。
    train, test = pd.read_csv(folder/'train.csv',skipinitialspace=True), pd.read_csv(folder/'test.csv',skipinitialspace=True)
    for frame in (train,test):
        frame.columns = frame.columns.str.strip()
        for column in ('episode_id','scenario_id'):
            frame[column] = frame[column].str.strip()
    assert not set(train.episode_id) & set(test.episode_id)
    decode = lambda frame,key: np.stack(frame[key].map(json.loads))
    linear = LinearDynamics(exogenous_dim=2).fit(decode(train,'state'),decode(train,'action_applied'),
        decode(train,'exogenous_available_at_decision'),decode(train,'next_state'))
    models = dict(persistence=PersistenceDynamics(), linear=linear)
    training = json.loads((ROOT/'reports/week4/results/training/config.json').read_text())
    for run in training['runs']:
        models[f"mlp_seed{run['seed']}"] = ActionMLP.load(ROOT/run['checkpoint'])
    output = ROOT/'reports/week4/results/rollout'
    residual_config = ROOT/'reports/week4/results/training_residual/config.json'
    if residual_config.exists():
        for run in json.loads(residual_config.read_text())['runs']:
            models[f"residual_seed{run['seed']}"] = ActionMLP.load(ROOT/run['checkpoint'])
    output.mkdir(parents=True,exist_ok=True)
    rate_config = ROOT/'reports/week4/results/training_rate_1/config.json'
    if rate_config.exists():
        for run in json.loads(rate_config.read_text())['runs']:
            models[f"rate_seed{run['seed']}"] = ActionMLP.load(ROOT/run['checkpoint'])
    linear.save(output/'linear.npz')
    metrics, origins = [], None
    for name, model in models.items():
        result = evaluate_rollouts(model,test)
        if origins is not None:
            assert origins == result['origins']
        origins = result['origins']
        np.testing.assert_allclose(result['free'][:,0],result['teacher'][:,0])
        np.savez_compressed(output/f'{name}_predictions.npz',**{key: result[key] for key in ('actual','free','teacher')})
        # terminal 为第 H 步误差，prefix 为前 H 步平均误差，温度单位均为 °C。
        for mode in ('free','teacher'):
            for h in (1,5,10,20):
                for definition in ('terminal','prefix'):
                    for index,variable in enumerate(('air','wall')):
                        actual = result['actual'][:,:h,index]
                        prediction = result[mode][:,:h,index]
                        if definition == 'terminal':
                            actual, prediction = actual[:,-1:],prediction[:,-1:]
                        metrics.append(dict(model=name,mode=mode,horizon=h,definition=definition,
                            variable=variable,origins=len(origins),mae_c=mae(actual,prediction),
                            rmse_c=rmse(actual,prediction)))
    pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(origins,columns=['episode_id','source_step']).to_csv(output/'origins.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(horizons=[1,5,10,20],
        origins=len(origins),units='Celsius',action='recorded future applied actions',
        exogenous='recorded contemporaneous observed outdoor and known load at each future step',
        interpretation='offline conditional dynamics evaluation; future exogenous sequence supplied externally',
        free='true state only at origin; recursive model predictions thereafter',
        teacher='actual state supplied at each step',overlapping_windows=True),indent=2))
    print(pd.DataFrame(metrics).query("mode == 'free' and definition == 'terminal' and variable == 'air'")[['model','horizon','rmse_c']].to_string(index=False))


if __name__ == '__main__':
    main()
