"""总说明：只用冻结训练块拟合尺度与模型，验证块选参数/最佳epoch，测试块不参与选择。"""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
import pandas as pd
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.boptest_dynamics import BoptestHistoryModel,exogenous_features


def episode_arrays(frame):
    """连续转移构成特征，当前动作对应当前到下一状态；输出能耗不回填到观测历史。"""
    x=np.stack(frame.state.map(json.loads));y=np.stack(frame.next_state.map(json.loads))
    weather=np.stack(frame.exogenous.map(json.loads));exo=exogenous_features(weather,frame.timestamp.to_numpy())
    features=np.column_stack([x,frame.action_applied_c.to_numpy(),exo])
    targets=np.column_stack([y,frame.energy_kwh.to_numpy()])
    return features,targets


def windows(frame,length,context):
    """历史可来自本回合上下文，但所有目标按完整时间块隔离。"""
    xs=[];ys=[]
    for _,episode in frame.groupby('episode_id',sort=False):
        episode=episode.sort_values('source_step');features,targets=episode_arrays(episode)
        assert np.array_equal(episode.source_step,np.arange(len(episode)))
        for k in range(max(length-1,context),len(episode)):
            xs.append(features[k-length+1:k+1]);ys.append(targets[k])
    return np.stack(xs),np.stack(ys)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--retrain',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text())
    source=ROOT/cfg['output']/'data';out=ROOT/cfg['output']/'training';out.mkdir(parents=True,exist_ok=True)
    if (out/'models.json').exists() and not args.retrain:
        saved=json.loads((out/'models.json').read_text())
        for split in ('train','val'):
            if hashlib.sha256((source/(split+'.csv')).read_bytes()).hexdigest()!=saved[split+'_sha256']:raise ValueError('Training source changed; use new output or explicit retrain')
        for spec in saved['models']:
            if hashlib.sha256((ROOT/spec['checkpoint']).read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Checkpoint changed')
        print('Verified existing training manifest/checkpoints; use --retrain for fresh optimization',flush=True);return
    train=pd.read_csv(source/'train.csv');val=pd.read_csv(source/'val.csv')
    assert set(train.episode_id).isdisjoint(val.episode_id)
    X,Y=windows(train,cfg['history_steps'],cfg['context_steps']);V,W=windows(val,cfg['history_steps'],cfg['context_steps'])
    mean=X.reshape(-1,12).mean(0);scale=np.maximum(X.reshape(-1,12).std(0),1e-6)
    ym=Y.mean(0);ys=np.maximum(Y.std(0),1e-6);A=((X-mean)/scale).reshape(len(X),-1);B=(Y-ym)/ys
    C=((V-mean)/scale).reshape(len(V),-1);D=(W-ym)/ys
    records=[];logs=[]
    def configure(kind,seed):
        model=BoptestHistoryModel(kind,cfg['history_steps'],seed)
        model.mean=mean;model.scale=scale;model.ymean=ym;model.yscale=ys;return model
    model=configure('arx',42);design=np.column_stack([A,np.ones(len(A))]);vdesign=np.column_stack([C,np.ones(len(C))])
    best=None
    for alpha in (.01,1.,100.):
        penalty=np.eye(design.shape[1])*alpha;penalty[-1,-1]=0
        coef=np.linalg.solve(design.T@design+penalty,design.T@B)
        loss=float(np.mean((vdesign@coef-D)**2));logs.append(dict(model='arx',alpha=alpha,val_loss=loss))
        if best is None or loss<best[0]:best=(loss,alpha,coef)
    model.coef=best[2];model.save(out/'arx.pt');records.append(dict(name='arx',kind='arx',checkpoint=str((out/'arx.pt').relative_to(ROOT)),val_loss=best[0],alpha=best[1]))
    Xtensor=torch.tensor((X-mean)/scale,dtype=torch.float32);Ytensor=torch.tensor(B,dtype=torch.float32)
    Vtensor=torch.tensor((V-mean)/scale,dtype=torch.float32);Wtensor=torch.tensor(D,dtype=torch.float32)
    for kind in ('mlp','gru'):
        for seed in cfg['training_seeds']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=configure(kind,seed)
            optimizer=torch.optim.Adam(model.network.parameters(),lr=.001);bestloss=np.inf;patience=0;weights=None
            for epoch in range(120):
                model.network.train();losses=[]
                for idx in np.array_split(rng.permutation(len(X)),max(1,int(np.ceil(len(X)/128)))):
                    optimizer.zero_grad();loss=((model.network(Xtensor[idx])-Ytensor[idx])**2).mean()
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.network.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
                model.network.eval()
                with torch.no_grad():vl=float(((model.network(Vtensor)-Wtensor)**2).mean())
                name=f'{kind}_seed{seed}';logs.append(dict(model=name,epoch=epoch,train_loss=float(np.mean(losses)),val_loss=vl))
                if vl<bestloss-1e-6:bestloss=vl;weights=copy.deepcopy(model.network.state_dict());bestepoch=epoch;patience=0
                else:patience+=1
                if patience>=15:break
            model.network.load_state_dict(weights);path=out/(name+'.pt');model.save(path)
            records.append(dict(name=name,kind=kind,seed=seed,checkpoint=str(path.relative_to(ROOT)),val_loss=bestloss,best_epoch=bestepoch))
            print(f'{name}: validation scaled MSE {bestloss:.5f}, epoch {bestepoch}',flush=True)
    persistence=BoptestHistoryModel('persistence',cfg['history_steps']);persistence.save(out/'persistence.pt')
    records.insert(0,dict(name='persistence',kind='persistence',checkpoint=str((out/'persistence.pt').relative_to(ROOT))))
    for row in records:row['sha256']=hashlib.sha256((ROOT/row['checkpoint']).read_bytes()).hexdigest()
    (out/'models.json').write_text(json.dumps(dict(models=records,train_episodes=sorted(train.episode_id.unique()),val_episodes=sorted(val.episode_id.unique()),
        train_sha256=hashlib.sha256((source/'train.csv').read_bytes()).hexdigest(),val_sha256=hashlib.sha256((source/'val.csv').read_bytes()).hexdigest(),
        normalization='training windows only; equal scaled MSE across 5 observations and interval energy',config=cfg),indent=2))
    pd.DataFrame(logs).to_csv(out/'training_log.csv',index=False)


if __name__=='__main__':main()
