"""在同一测试起点评估输出速率约束，保存逐步原输出和修正后输出。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.action_mlp import ActionMLP
from models.physics_constraints import RateConstrainedDynamics
from metrics.prediction import mae,rmse


def main():
    output=ROOT/'reports/week4/results/physics_constraints'
    output.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'data/processed/week4_thermal/test.csv',skipinitialspace=True)
    data.columns=data.columns.str.strip()
    results=[]
    for family in ('training','training_residual','training_rate_1'):
        config=json.loads((ROOT/f'reports/week4/results/{family}/config.json').read_text())
        for run in config['runs']:
            model=ActionMLP.load(ROOT/run['checkpoint'])
            constrained=RateConstrainedDynamics(model)
            truth,raw_all,projected_all,flags=[],[],[],[]
            origins=[]
            for episode_id,ep in data.groupby('episode_id'):
                ep=ep.sort_values('source_step')
                decode=lambda key:np.stack(ep[key].map(json.loads))
                x,u,d,y=map(decode,('state','action_applied','exogenous_available_at_decision','next_state'))
                starts=np.arange(len(ep)-19)
                predicted=x[starts].copy()
                for k in range(20):
                    indices=starts+k
                    predicted,detail=constrained.predict_with_diagnostics(predicted,u[indices],d[indices])
                    truth.append(y[indices])
                    raw_all.append(detail['raw'])
                    projected_all.append(predicted.copy())
                    flags.append(detail['triggered'])
                    origins.extend((str(episode_id).strip(),int(t),k+1) for t in starts)
            actual,raw,projected,triggered=map(np.concatenate,(truth,raw_all,projected_all,flags))
            label=f"{family}_seed{run['seed']}"
            np.savez_compressed(output/f'{label}.npz',actual=actual,raw=raw,projected=projected,triggered=triggered)
            for index,name in enumerate(('air','wall')):
                results.append(dict(model=label,variable=name,predictions=len(actual),
                    raw_mae=mae(actual[:,index:index+1],raw[:,index:index+1]),
                    projected_mae=mae(actual[:,index:index+1],projected[:,index:index+1]),
                    raw_rmse=rmse(actual[:,index:index+1],raw[:,index:index+1]),
                    projected_rmse=rmse(actual[:,index:index+1],projected[:,index:index+1]),
                    trigger_count=int(triggered[:,index].sum()),violation_rate=float(triggered[:,index].mean())))
    pd.DataFrame(origins,columns=['episode_id','source_step','horizon']).to_csv(output/'origins.csv',index=False)
    pd.DataFrame(results).to_csv(output/'metrics.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False))
    (output/'config.json').write_text(json.dumps(dict(outdoor_bounds_c=[-5,17],
        noise_margin_c=.03,method='output delta projection',horizon=20,
        evaluation='raw and projected outputs share the constrained rollout input state at every step',
        uncertainty='3-sigma margin is a heuristic; Gaussian noise has unbounded support'),indent=2))


if __name__=='__main__':
    main()
