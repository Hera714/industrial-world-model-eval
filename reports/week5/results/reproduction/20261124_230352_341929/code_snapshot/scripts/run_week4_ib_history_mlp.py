"""IB 部分观测对照：独立回合采集、当前观测 MLP 与历史动作 GRU。"""
from pathlib import Path
import sys
import json
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
from torch import nn
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulators.industrial_benchmark import IndustrialBenchmarkEnv


class HistoryPredictor(nn.Module):
    """输出下一公开观测增量；GRU 输入按时间排列的观测和该步动作。"""
    def __init__(self, history):
        super().__init__()
        self.history=history
        self.encoder=nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(20*9,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.head=nn.Linear(64,6)

    def forward(self,x):
        features=self.encoder(x)
        return self.head(features)


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    out=ROOT/'reports/week4/results/ib_history'
    out.mkdir(parents=True,exist_ok=True)
    with np.load(out/'episodes.npz') as existing:
        existing_observations=existing['observations'].copy()
        existing_actions=existing['actions'].copy()
    # 复用原始轨迹，保持环境样本和分组切分完全一致。
    observations,actions=existing_observations,existing_actions
    ids=np.random.default_rng(2026).permutation(30)
    splits=dict(train=ids[:21].tolist(),val=ids[21:25].tolist(),test=ids[25:].tolist())
    history=20
    def windows(episode_ids):
        features=[]; targets=[]; origins=[]
        for ep in episode_ids:
            for t in range(history-1,300):
                features.append(np.concatenate([observations[ep,t-history+1:t+1],actions[ep,t-history+1:t+1]],axis=1))
                targets.append(observations[ep,t+1]-observations[ep,t])
                origins.append((ep,t))
        return np.array(features),np.array(targets),origins
    data={key:windows(value) for key,value in splits.items()}
    # 所有缩放参数仅用训练 episode；目标为公开观测增量。
    xmean=data['train'][0].mean(axis=(0,1)); xscale=np.maximum(data['train'][0].std(axis=(0,1)),1e-6)
    ymean=data['train'][1].mean(axis=0); yscale=np.maximum(data['train'][1].std(axis=0),1e-6)
    tensors={key:(torch.tensor((x-xmean)/xscale,dtype=torch.float32,device=device),
                  torch.tensor((y-ymean)/yscale,dtype=torch.float32,device=device)) for key,(x,y,_) in data.items()}
    metrics=[]; logs=[]; best_runs=[]
    for use_history in (True,):
        name='history_mlp'
        for seed in (42,43,44):
            torch.manual_seed(seed)
            model=HistoryPredictor(use_history).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            rng=np.random.default_rng(seed)
            best=float('inf'); stale=0
            for epoch in range(1,101):
                model.train()
                x,y=tensors['train']
                for batch in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/256))):
                    loss=nn.functional.mse_loss(model(x[batch]),y[batch])
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad():
                    val=float(nn.functional.mse_loss(model(tensors['val'][0]),tensors['val'][1]))
                logs.append(dict(model=name,seed=seed,epoch=epoch,val_standardized_mse=val))
                if val<best:
                    best=val; stale=0; best_epoch=epoch
                    torch.save(dict(weights=model.state_dict(),history=use_history,
                        xmean=xmean.tolist(),xscale=xscale.tolist(),ymean=ymean.tolist(),yscale=yscale.tolist()),out/f'{name}_{seed}.pt')
                else:
                    stale+=1
                if stale>=12:
                    break
            checkpoint=torch.load(out/f'{name}_{seed}.pt',map_location=device,weights_only=True)
            model.load_state_dict(checkpoint['weights'])
            model.eval()
            with torch.no_grad():
                delta=model(tensors['test'][0]).cpu().numpy()*yscale+ymean
            current=data['test'][0][:,-1,:6]
            actual=current+data['test'][1]
            predicted=current+delta
            np.savez_compressed(out/f'{name}_{seed}_predictions.npz',actual=actual,predicted=predicted)
            best_runs.append(dict(model=name,seed=seed,best_epoch=best_epoch,val_standardized_mse=best))
            for index,variable in enumerate(('setpoint','velocity','gain','shift','fatigue','consumption')):
                error=predicted[:,index]-actual[:,index]
                metrics.append(dict(model=name,seed=seed,variable=variable,mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2)))))
            print(name,seed,best_epoch,flush=True)
    original_config=json.loads((out/'config.json').read_text())
    assert splits == original_config['splits']
    expected=pd.read_csv(out/'test_origins.csv').to_numpy()
    np.testing.assert_array_equal(expected,np.array(data['test'][2]))
    pd.DataFrame(metrics).to_csv(out/'history_mlp_metrics.csv',index=False)
    pd.DataFrame(logs).to_csv(out/'history_mlp_training_log.csv',index=False)
    (out/'history_mlp_config.json').write_text(json.dumps(dict(splits=splits,environment_seeds=list(range(7000,7030)),
        action_seeds=list(range(17000,17030)),history=history,episode_steps=300,setpoint=50,
        device=device,runs=best_runs,objective='next public observation delta; standardized MSE',
        input='20 observed states each paired with its outgoing action, including current candidate action',
        split_seed=2026,units='IB benchmark scales',learning_rate=.001,max_epochs=100,patience=12),indent=2))
    print(pd.DataFrame(metrics).groupby(['model','variable']).rmse.mean().to_string(),flush=True)


if __name__=='__main__':
    main()
