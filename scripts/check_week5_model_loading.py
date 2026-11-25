"""检查统一加载后的形状、物理尺度和窗口隔离；复用已有检查点，不训练模型。"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.dynamics_base import load_dynamics


def main():
    """热工模型检查单样本/批量一致性，IB 逐项核对既有预测和独立历史。"""
    config = yaml.safe_load((ROOT/'configs/experiments/week5_evaluation.yaml').read_text())
    results = []
    for spec in config['models']:
        model = load_dynamics(spec, ROOT, hashes=config.get('sha256'))
        model.reset()
        x = np.array([20., 19.]); u = np.array([2.]); d = np.array([5., .3])
        single = model.predict_next(x, u, d)
        batch = model.predict_next(np.stack([x, x]), u, d)
        np.testing.assert_allclose(batch, np.stack([single, single]), atol=1e-5)
        try:
            model.predict_next([np.nan, 19.], u, d)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite state accepted')
        results.append(dict(model=spec['name'], shape_and_finite_checks='passed'))

    directory = ROOT/'reports/week4/results/ib_history'
    data = np.load(directory/'episodes.npz')
    origins = pd.read_csv(directory/'test_origins.csv')
    windows = np.array([np.concatenate([
        data['observations'][int(row.episode), int(row.source_step)-19:int(row.source_step)+1],
        data['actions'][int(row.episode), int(row.source_step)-19:int(row.source_step)+1]
    ], axis=-1) for row in origins.itertuples()])
    for name in ('mlp', 'history_mlp', 'gru'):
        for seed in (42, 43, 44):
            model = load_dynamics(dict(kind='ib_'+name, checkpoint=str(directory/f'{name}_{seed}.pt')), ROOT)
            predicted = model.predict_window(windows)
            expected = np.load(directory/f'{name}_{seed}_predictions.npz')['predicted']
            # 使用原始网络在同一设备复核；旧文件可能来自 GPU，差异单独记录。
            if name == 'history_mlp':
                from scripts.run_week4_ib_history_mlp import HistoryPredictor
            else:
                from scripts.run_week4_ib_history import HistoryPredictor
            original = HistoryPredictor(name == 'gru').eval()
            checkpoint = torch.load(directory/f'{name}_{seed}.pt', weights_only=True, map_location='cpu')
            original.load_state_dict(checkpoint['weights'])
            with torch.no_grad():
                reference = original(torch.tensor((windows-model.xmean)/model.xscale, dtype=torch.float32)).numpy()
            reference = windows[:, -1, :6]+reference*model.yscale+model.ymean
            np.testing.assert_allclose(predicted, reference, atol=1e-6)
            # A、B、A 顺序调用应还原同一窗口预测，不能沿用 B 的隐藏状态。
            a = model.predict_window(windows[0])
            model.predict_window(windows[1])
            np.testing.assert_allclose(model.predict_window(windows[0]), a, atol=1e-5)
            model.reset(observation_history=windows[0, :19, :6], action_history=windows[0, :19, 6:])
            np.testing.assert_allclose(model.predict_next(windows[0, -1, :6], windows[0, -1, 6:]), a, atol=1e-5)
            model.reset()
            if name != 'mlp':
                try:
                    model.predict_next(windows[0, -1, :6], windows[0, -1, 6:])
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('Missing history accepted')
            results.append(dict(model=f'ib_{name}_{seed}', windows=len(windows),
                                max_saved_prediction_difference=float(np.abs(predicted-expected).max()),
                                independent_windows='passed'))
    output = ROOT/'reports/week5/results/model_loading_checks.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f'Passed {len(results)} model loading checks; {output}')


if __name__ == '__main__':
    main()
