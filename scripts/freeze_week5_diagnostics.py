"""冻结动作反事实计划和工况分解规则，保留原评测配置与检查点。"""
from pathlib import Path
import hashlib
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'configs/experiments/week5_diagnostics.yaml'
if target.exists(): raise FileExistsError('Diagnostic config already exists; do not silently overwrite it')
source=ROOT/'configs/experiments/week5_evaluation.yaml'
protocol=yaml.safe_load(source.read_text()); horizon=80
plans={'constant_0':np.zeros(horizon),'constant_1':np.ones(horizon),
    'constant_2p5':np.full(horizon,2.5),'constant_5':np.full(horizon,5.),
    'step_up':np.r_[np.zeros(12),np.full(horizon-12,5.)],
    'step_down':np.r_[np.full(12,5.),np.zeros(horizon-12)],
    'pulse':np.r_[np.zeros(12),np.full(12,5.),np.zeros(horizon-24)],
    'ramp':np.linspace(0,5,horizon)}
folder=ROOT/protocol['output']/'rollout'
paths=[source,folder/'origins.csv',folder/'failures.csv',folder/'validity_by_origin.csv']
paths.extend(folder/f'{spec["name"]}_predictions.npz' for spec in protocol['models'])
config=dict(protocol_version='1.0-action-and-error-diagnostics',evaluation=protocol,
    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    action_experiment=dict(horizon=80,initial_states=[[18.,19.],[20.,19.],[22.,21.]],
        weather_profiles=['constant','daily'],outdoor_mean_c=6.,outdoor_amplitude_c=4.,load_kw=.3,
        environment_seed=42,process_noise_std=0.,measurement_noise_std=0.,weather_noise_std=0.,
        plans={name:values.tolist() for name,values in plans.items()},
        references={name:('constant_5' if name=='step_down' else 'constant_0') for name in plans if name!='constant_0'},
        onset='first action index differing from reference',direction_epsilon_c=.01,lag_fraction=.5,
        lag='first crossing of 50% simulator paired peak; fixed threshold shared by all models',
        example_initial=[20.,19.],example_plan='step_up'),
    decomposition=dict(horizons=protocol['horizons'],power_bins=['<1','[1,4)','>=4'],
        power_definition='applied action at source_step+horizon-1, the transition producing target state',
        initial_air_bins=['<19','[19,21)','>=21'],initial_definition='episode initial true air temperature',
        large_abs_error_c=.5,case_selection='retain all flagged cases; worst case per model/mode/condition/variable/horizon'),
    output='reports/week5/results/diagnostics')
target.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
print(target)
