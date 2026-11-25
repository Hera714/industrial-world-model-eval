"""总说明：补训历史残差MLP，沿用原训练/验证窗口和优化配置，原始八模型清单保持可追溯。"""
from pathlib import Path
import sys,json,copy,hashlib
import numpy as np
import pandas as pd
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.boptest_dynamics import BoptestHistoryModel
from scripts.train_week6 import windows


def main():
    torch.set_num_threads(2)
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text())
    base=ROOT/cfg['output'];out=base/'residual_mlp';out.mkdir(exist_ok=True)
    if (out/'models.json').exists():
        saved=json.loads((out/'models.json').read_text())
        for split,key in [('train','training_data_sha256'),('val','validation_data_sha256')]:
            assert hashlib.sha256((base/'data'/(split+'.csv')).read_bytes()).hexdigest()==saved[key]
        for model in saved['models']:
            assert hashlib.sha256((ROOT/model['checkpoint']).read_bytes()).hexdigest()==model['sha256']
        print('Existing residual checkpoints and training inputs verified',flush=True);return
    X,Y=windows(pd.read_csv(base/'data/train.csv'),16,96)
    V,W=windows(pd.read_csv(base/'data/val.csv'),16,96)
    Y[:,:5]-=X[:,-1,:5];W[:,:5]-=V[:,-1,:5]
    mean=X.reshape(-1,12).mean(0);scale=np.maximum(X.reshape(-1,12).std(0),1e-6)
    ym=Y.mean(0);ys=np.maximum(Y.std(0),1e-6)
    xt=torch.tensor((X-mean)/scale,dtype=torch.float32);yt=torch.tensor((Y-ym)/ys,dtype=torch.float32)
    vt=torch.tensor((V-mean)/scale,dtype=torch.float32);wt=torch.tensor((W-ym)/ys,dtype=torch.float32)
    records=[];logs=[]
    for seed in (42,43,44):
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=BoptestHistoryModel('residual_mlp',16,seed)
        model.mean=mean;model.scale=scale;model.ymean=ym;model.yscale=ys
        optimizer=torch.optim.Adam(model.network.parameters(),lr=.001)
        best=np.inf;patience=0
        for epoch in range(120):
            model.network.train();losses=[]
            for idx in np.array_split(rng.permutation(len(X)),int(np.ceil(len(X)/128))):
                optimizer.zero_grad();loss=((model.network(xt[idx])-yt[idx])**2).mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(model.network.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            model.network.eval()
            with torch.no_grad():vl=float(((model.network(vt)-wt)**2).mean())
            logs.append(dict(seed=seed,epoch=epoch,train_loss=np.mean(losses),validation_loss=vl))
            if vl<best-1e-6:best=vl;weights=copy.deepcopy(model.network.state_dict());bestepoch=epoch;patience=0
            else:patience+=1
            if patience>=15:break
        model.network.load_state_dict(weights);name=f'residual_mlp_seed{seed}';path=out/(name+'.pt');model.save(path)
        np.testing.assert_allclose(model.predict(V[:2]),BoptestHistoryModel.load(path).predict(V[:2]))
        records.append(dict(name=name,kind='residual_mlp',seed=seed,checkpoint=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),best_epoch=bestepoch,val_loss=best))
        print(name,best,bestepoch,flush=True)
    (out/'models.json').write_text(json.dumps(dict(models=records,config=cfg,architecture='Flatten192 -> Linear128 ReLU -> Linear64 ReLU -> Linear6; first5 residual, energy direct',
        training_data_sha256=hashlib.sha256((base/'data/train.csv').read_bytes()).hexdigest(),validation_data_sha256=hashlib.sha256((base/'data/val.csv').read_bytes()).hexdigest()),indent=2))
    pd.DataFrame(logs).to_csv(out/'training_log.csv',index=False)


if __name__=='__main__':main()
