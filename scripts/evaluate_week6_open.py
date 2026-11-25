"""总说明：冻结 BOPTEST 模型的统一开环评测、天气/噪声/缺失矩阵，保留每个起点原始输出。"""
from pathlib import Path
import sys,json,hashlib,os
import numpy as np
import pandas as pd
import yaml
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.boptest_dynamics import BoptestHistoryModel
from scripts.train_week6 import episode_arrays

CONDITIONS=['ideal_weather_reference','persistence_weather','noise_0p1','noise_0p3','missing_1','missing_4']


def rollout(model,features,start,horizon,condition,mode='free_running'):
    """模型历史独立复制；自由递推不读未来观测，历史缺失仅用过去有效值填补。"""
    length=model.length;history=features[start-length+1:start+1].copy();pred=[];warning=[]
    rng=np.random.default_rng(64042+start)
    if condition.startswith('noise_'):
        sigma=.1 if condition=='noise_0p1' else .3;history[:,:3]+=rng.normal(0,sigma,size=(length,3))
    burst=int(condition.split('_')[-1]) if condition.startswith('missing_') else 0
    last_supply=history[-2,1] if length>1 else history[-1,1]
    if burst:history[-1,1]=last_supply
    weather=features[start,6:8].copy()
    for k in range(horizon):
        try:value=model.predict(history)
        except Exception as exc:return pred,warning,dict(step=k+1,reason=type(exc).__name__+':'+str(exc))
        if np.max(np.abs(value))>1000:return pred,warning,dict(step=k+1,reason='divergence')
        physical=not (0<=value[0]<=50 and np.all((value[1:3]>=0)&(value[1:3]<=90)) and value[4]>=0 and value[5]>=0)
        pred.append(value.copy());warning.append(physical)
        if k+1==horizon:break
        nextrow=features[start+k+1].copy()
        if mode=='free_running':nextrow[:5]=value[:5]
        elif condition.startswith('noise_'):
            nextrow[:3]+=rng.normal(0,sigma,size=3)
        if condition!='ideal_weather_reference':nextrow[6:8]=weather
        if burst and k+1<burst:nextrow[1]=last_supply
        history=np.vstack([history[1:],nextrow])
    return pred,warning,None


def summarize(points):
    """先在独立时间块内聚合；模型训练种子保留，窗口不当作独立重复。"""
    keys=['model','family','training_seed','split','condition','mode','horizon','variable','unit','episode_id']
    episode=points.groupby(keys,dropna=False).agg(requested=('valid','size'),valid=('valid','sum'),
        mae=('abs_error','mean'),mse=('squared_error','mean'),physical_warning=('physical_warning','sum')).reset_index()
    episode['rmse']=np.sqrt(episode.pop('mse'));episode['valid_fraction']=episode.valid/episode.requested
    summary=episode.groupby(keys[:-1],dropna=False).agg(mae_mean=('mae','mean'),mae_episode_sd=('mae','std'),
        rmse_mean=('rmse','mean'),rmse_episode_sd=('rmse','std'),episodes=('episode_id','nunique'),
        valid_fraction=('valid_fraction','mean'),requested=('requested','sum'),valid=('valid','sum')).reset_index()
    return episode,summary


def main():
    torch.set_num_threads(2);cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text())
    base=ROOT/cfg['output'];manifest=json.loads(Path(os.environ.get('WEEK6_MODELS_MANIFEST',base/'training/models.json')).read_text());output=Path(os.environ.get('WEEK6_OPEN_OUTPUT',base/'open_loop'));output.mkdir(parents=True,exist_ok=True)
    records=[];identities=[];failures=[];artifacts={};bounds=[]
    train=pd.read_csv(base/'data/train.csv');trainweather=np.stack(train.exogenous.map(json.loads));low=trainweather.min(0);high=trainweather.max(0)
    for split in ('id_test','ood_cold','ood_warm'):
        frame=pd.read_csv(base/'data'/f'{split}.csv')
        for eid,episode in frame.groupby('episode_id',sort=False):
            f,y=episode_arrays(episode);weather=np.stack(episode.exogenous.map(json.loads));mask=episode.source_step.to_numpy()>=cfg['context_steps']
            bounds.append(dict(episode_id=eid,split=split,outdoor_mean_c=weather[mask,0].mean(),outdoor_min_c=weather[mask,0].min(),outdoor_max_c=weather[mask,0].max(),
                train_outdoor_min_c=low[0],train_outdoor_max_c=high[0],outdoor_outside_training_fraction=np.mean((weather[mask,0]<low[0])|(weather[mask,0]>high[0]))))
            origins=list(range(cfg['context_steps'],len(f)-max(cfg['horizons'])+1,4))
            for spec in manifest['models']:
                assert hashlib.sha256((ROOT/spec['checkpoint']).read_bytes()).hexdigest()==spec['sha256']
                model=BoptestHistoryModel.load(ROOT/spec['checkpoint'])
                for condition in CONDITIONS:
                    for mode in ('teacher_forcing','free_running'):
                        raw=np.full((len(origins),max(cfg['horizons']),6),np.nan)
                        for index,start in enumerate(origins):
                            values,warnings,failure=rollout(model,f,start,max(cfg['horizons']),condition,mode)
                            if values:raw[index,:len(values)]=values
                            if failure:failures.append(dict(model=spec['name'],episode_id=eid,start=start,condition=condition,mode=mode,**failure))
                            for h in cfg['horizons']:
                                truth=y[start+h-1];prediction=raw[index,h-1]
                                for v,(name,unit) in enumerate(zip(cfg['state_names']+['interval_energy'],['C','C','C','kW','kW','kWh'])):
                                    error=prediction[v]-truth[v]
                                    records.append(dict(model=spec['name'],family=spec['kind'],training_seed=spec.get('seed',-1),split=split,episode_id=eid,start=start,
                                        condition=condition,mode=mode,horizon=h,variable=name,unit=unit,actual=truth[v],prediction=prediction[v],
                                        abs_error=abs(error),squared_error=error**2,valid=bool(np.isfinite(prediction[v])),physical_warning=warnings[h-1] if len(warnings)>=h else False))
                        artifacts[f'{eid}__{spec["name"]}__{condition}__{mode}']=raw
            print('evaluated '+eid,flush=True)
    points=pd.DataFrame(records);points.to_csv(output/'endpoint_errors.csv.gz',index=False)
    ep,summary=summarize(points);ep.to_csv(output/'metrics_by_episode.csv',index=False);summary.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(bounds).to_csv(output/'weather_support.csv',index=False)
    (output/'failures.json').write_text(json.dumps(failures,indent=2));np.savez_compressed(output/'raw_predictions.npz',**artifacts)
    # 每变量分别选案例，避免把不同物理单位的误差当成统一尺度排名。
    points.groupby(['variable','condition','mode'],group_keys=False).apply(lambda g:g.nlargest(3,'abs_error'),include_groups=False).to_csv(output/'failure_cases_by_variable.csv')


if __name__=='__main__':main()
