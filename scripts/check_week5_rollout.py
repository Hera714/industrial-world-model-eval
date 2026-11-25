"""核对长时序原始输出：共同起点、两模式首步一致，以及与单步评测一致。"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'reports/week5/results'
origins = pd.read_csv(folder/'rollout/origins.csv')
single = pd.read_csv(folder/'one_step/predictions.csv.gz')
for path in (folder/'rollout').glob('*_predictions.npz'):
    name = path.name.removesuffix('_predictions.npz')
    with np.load(path) as values:
        reference = values['recorded_observation__free_running'][:, 0]
        for condition in ('recorded_observation', 'persistence_forecast'):
            for mode in ('teacher_forcing', 'free_running'):
                np.testing.assert_allclose(values[condition+'__'+mode][:, 0], reference, atol=1e-5)
        for index, variable in enumerate(('air', 'wall')):
            rows = single[(single.model == name) & (single.variable == variable)]
            paired = origins.merge(rows, on=['episode_id', 'source_step'], validate='one_to_one')
            np.testing.assert_allclose(paired.predicted, reference[:, index], atol=1e-5)
print('All 8 model outputs agree at horizon 1 across modes, weather conditions and single-step evaluation.')
