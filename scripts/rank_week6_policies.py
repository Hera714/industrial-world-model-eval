"""总说明：将实际闭环产生的前20步动作冻结为开环计划，配对比较模型策略排序与模拟器已执行成本。"""
from pathlib import Path
import sys,json,itertools
import numpy as np
import pandas as pd
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.boptest_dynamics import BoptestHistoryModel,exogenous_features
from controllers.boptest_mpc import stage_cost


def predict_plan(model,source,config,horizon=20):
    """从同一真实历史开始，未来输入只使用冻结动作、已知舒适日程和起点天气。"""
    start=config['context_steps'];L=config['history_steps'];history=[]
    for row in source.iloc[start-L+1:start+1].itertuples():
        history.append(np.r_[json.loads(row.true_before),row.applied_action_c,
            exogenous_features(np.asarray(json.loads(row.weather)),np.asarray(row.timestamp))])
    history=np.stack(history);weather=history[-1,6:8].copy();previous=float(source.applied_action_c.iloc[start-1]);cost=0.;values=[]
    for k,row in enumerate(source.iloc[start:start+horizon].itertuples()):
        history[-1,5]=row.applied_action_c
        try:prediction=model.predict(history)
        except Exception as exc:return np.nan,values,type(exc).__name__
        if not np.isfinite(prediction).all() or np.max(np.abs(prediction))>1000:return np.nan,values,'invalid_rollout'
        if prediction[5]<0:return np.nan,values,'negative_energy'
        p=config['control'];cost+=float(stage_cost(prediction[0],prediction[5],row.lower_c,row.upper_c,previous,row.applied_action_c,
            comfort_weight=p['comfort_weight'],energy_weight=p['energy_weight'],change_weight=p['action_change_weight']))
        values.append(prediction.tolist());previous=row.applied_action_c
        nextrow=np.r_[prediction[:5],previous,exogenous_features(weather,np.asarray(row.timestamp+900))]
        history=np.vstack([history[1:],nextrow])
    return cost,values,''


def main():
    torch.set_num_threads(2);cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text());base=ROOT/cfg['output']
    models=json.loads((base/'training/models.json').read_text())['models'];out=base/'policy_ranking';out.mkdir(parents=True,exist_ok=True)
    plans={};records=[];raw={};rankings=[]
    for episode in cfg['episodes']:
        if episode['split'] in ('train','val'):continue
        eid=episode['episode_id'];plans={}
        for path in (base/'closed_loop').glob(eid+'__*__clean.csv'):
            frame=pd.read_csv(path);plans[frame.model.iloc[0]]=frame
        if len(plans)!=len(models)+1:raise ValueError('Incomplete policy set '+eid)
        initial=[json.loads(f.true_before.iloc[cfg['context_steps']]) for f in plans.values()]
        for state in initial[1:]:np.testing.assert_allclose(state,initial[0],atol=1e-5)
        for spec in models:
            model=BoptestHistoryModel.load(ROOT/spec['checkpoint']);current=[]
            for policy,frame in plans.items():
                predicted,values,reason=predict_plan(model,frame,cfg)
                truth=float(frame.iloc[cfg['context_steps']:cfg['context_steps']+20].cost.sum())
                row=dict(episode_id=eid,judge_model=spec['name'],policy=policy,horizon=20,true_cost=truth,predicted_cost=predicted,
                    valid=np.isfinite(predicted),failure_reason=reason)
                records.append(row);current.append(row);raw[eid+'__'+spec['name']+'__'+policy]=values
            concordant=discordant=tied=invalid=0
            for a,b in itertools.combinations(current,2):
                if not a['valid'] or not b['valid']:invalid+=1;continue
                actual=a['true_cost']-b['true_cost'];predicted=a['predicted_cost']-b['predicted_cost']
                if abs(actual)<1e-6 or abs(predicted)<1e-6:tied+=1
                elif np.sign(actual)==np.sign(predicted):concordant+=1
                else:discordant+=1
            valid=[r for r in current if r['valid']]
            selected=min(valid,key=lambda r:r['predicted_cost']) if valid else None
            rankings.append(dict(episode_id=eid,judge_model=spec['name'],concordant=concordant,discordant=discordant,tied=tied,invalid_pairs=invalid,
                direction_agreement=concordant/(concordant+discordant) if concordant+discordant else np.nan,
                valid_policies=len(valid),requested_policies=len(current),selected_policy=selected['policy'] if selected else None,
                selected_true_cost=selected['true_cost'] if selected else np.nan,
                regret_vs_all_true_policies=selected['true_cost']-min(r['true_cost'] for r in current) if selected else np.nan))
    pd.DataFrame(records).to_csv(out/'plan_costs.csv',index=False);pd.DataFrame(rankings).to_csv(out/'ranking_by_episode.csv',index=False)
    (out/'raw_predictions.json').write_text(json.dumps(raw))
    (out/'protocol.json').write_text(json.dumps(dict(horizon=20,weather='persistence at origin',
        semantics='Evaluate frozen realized action sequences, not counterfactual feedback policies. True costs are the same simulator trajectory prefixes.',
        invalid='Negative energy/failed model rollout stays invalid; report validity and regret jointly; no invented score.',
        ties='Absolute cost difference <=1e-6, omitted from sign agreement but counted.'),indent=2))


if __name__=='__main__':main()
