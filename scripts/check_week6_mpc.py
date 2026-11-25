"""总说明：通过公共工厂检查11个检查点的MPC接入，并逐步核对规划与实际评测代价定义。"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from controllers.mpc import create_boptest_mpc
from controllers.boptest_mpc import stage_cost
from scripts.train_week6 import episode_arrays


def main():
    import torch
    torch.set_num_threads(2)
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text());base=ROOT/cfg['output']
    models=json.loads((base/'training/models.json').read_text())['models']+json.loads((base/'residual_mlp/models.json').read_text())['models']
    frame=pd.read_csv(base/'data/id_test_day160.csv');features,_=episode_arrays(frame);history=features[81:97]
    records=[]
    for spec in models:
        controller=create_boptest_mpc(spec,cfg,ROOT)
        prediction=controller.model.predict(history)
        assert prediction.shape==(6,) and np.isfinite(prediction).all()
        np.testing.assert_allclose(prediction,controller.model.predict(history[None])[0],rtol=1e-5,atol=1e-5)
        action,info=controller.act(history,int(frame.timestamp.iloc[96]),np.full(12,21.),np.full(12,24.),96,float(features[95,5]))
        assert not info['error']
        assert (action is None and info['fallback_reason'] in ('no_valid_candidates','timeout')) or (action is not None and 18<=action[0]<=24)
        records.append(dict(model=spec['name'],output_shape=list(prediction.shape),action=action,fallback=info['fallback_reason'],decision_time_s=info['decision_time_s']))
    scored=0;episodes=0
    for path in (base/'closed_loop').glob('*.csv'):
        if path.name.endswith('.partial.csv'):continue
        trace=pd.read_csv(path)
        if 'true_before' not in trace:continue
        previous=trace.applied_action_c.shift(1);part=trace[~trace.context]
        expected=stage_cost(part.zone_c.to_numpy(),part.energy_kwh.to_numpy(),part.lower_c.to_numpy(),part.upper_c.to_numpy(),
            previous.loc[part.index].to_numpy(),part.applied_action_c.to_numpy(),comfort_weight=1.,energy_weight=1.,change_weight=.02)
        np.testing.assert_allclose(expected,part.cost,rtol=1e-10,atol=1e-10)
        scored+=len(part);episodes+=1
    assert len(records)==11 and episodes==204
    out=base/'mpc';out.mkdir(exist_ok=True)
    result=dict(status='passed',checkpoints=11,verified_episodes=episodes,verified_scored_steps=scored,models=records,
        known_equation_mpc='Not implemented for BOPTEST. Embedded controller is an engineering baseline, not an oracle dynamics MPC.',
        contract='predict(history[N,16,12])->[N,6]; predict_transition separates next observation and interval energy')
    (out/'interface_and_cost_verification.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(records).to_csv(out/'model_interface.csv',index=False)
    print(json.dumps({k:v for k,v in result.items() if k!='models'}))


if __name__=='__main__':main()
