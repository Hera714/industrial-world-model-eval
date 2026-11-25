"""训练动作条件 MLP：训练集标准化，验证集选宽度和早停，保存完整检查点。"""
import csv
import json
from pathlib import Path
import sys
import os
import argparse
# CUDA 初始化前设置，确保确定性矩阵运算可执行。
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.action_mlp import ActionMLP


def load_split(name):
    """只加载指定分组切分，字段按动力学规范解码。"""
    frame = pd.read_csv(ROOT/f'data/processed/week4_thermal/{name}.csv')
    x = np.stack(frame.state.map(json.loads))
    u = np.stack(frame.action_applied.map(json.loads))
    d = np.stack(frame.exogenous_available_at_decision.map(json.loads))
    y = np.stack(frame.next_state.map(json.loads))
    return np.concatenate([x,u,d],axis=1), y


def train(hidden, seed, features, targets, validation, val_targets, device, output, residual=False, rate_weight=0.0):
    """每个候选独立初始化；物理单位验证 MSE 决定最佳轮次。"""
    model = ActionMLP(hidden=hidden, seed=seed, device=device, residual=residual)
    # 残差模式仅改变学习目标；验证仍以物理单位下一状态误差评分。
    learning_targets = targets-features[:,:2] if residual else targets
    model.input_mean = features.mean(axis=0)
    model.input_scale = np.maximum(features.std(axis=0),1e-6)
    model.output_mean = learning_targets.mean(axis=0)
    model.output_scale = np.maximum(learning_targets.std(axis=0),1e-6)
    x = torch.tensor((features-model.input_mean)/model.input_scale,dtype=torch.float32,device=device)
    y = torch.tensor((learning_targets-model.output_mean)/model.output_scale,dtype=torch.float32,device=device)
    vx = torch.tensor((validation-model.input_mean)/model.input_scale,dtype=torch.float32,device=device)
    optimizer = torch.optim.Adam(model.network.parameters(),lr=.001)
    rng = np.random.default_rng(seed)
    # 物理增量边界与推理投影相同，正则梯度直接作用于原始网络输出。
    from models.physics_constraints import RateConstrainedDynamics
    from models.dynamics_base import PersistenceDynamics
    bounds_model = RateConstrainedDynamics(PersistenceDynamics())
    _, bounds = bounds_model.predict_with_diagnostics(features[:,:2],features[:,2:3],features[:,3:])
    lower = torch.tensor(bounds['low_delta'],dtype=torch.float32,device=device)
    upper = torch.tensor(bounds['high_delta'],dtype=torch.float32,device=device)
    current = torch.tensor(features[:,:2],dtype=torch.float32,device=device)
    mean = torch.tensor(model.output_mean,dtype=torch.float32,device=device)
    scale = torch.tensor(model.output_scale,dtype=torch.float32,device=device)
    best, stale, history = float('inf'), 0, []
    model.is_fitted = True
    path = output/f'mlp_hidden{hidden}_seed{seed}.pt'
    for epoch in range(1,301):
        model.network.train()
        total = 0.
        for indices in np.array_split(rng.permutation(len(x)), max(1,int(np.ceil(len(x)/256)))):
            prediction = model.network(x[indices])
            fit_loss = torch.nn.functional.mse_loss(prediction,y[indices])
            delta = prediction*scale+mean
            if not residual:
                delta = delta-current[indices]
            excess = torch.relu(lower[indices]-delta)+torch.relu(delta-upper[indices])
            penalty = torch.mean((excess/.03)**2)
            loss = fit_loss+rate_weight*penalty
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(indices)
        model.network.eval()
        with torch.no_grad():
            physical = model.network(vx).cpu().numpy()*model.output_scale+model.output_mean
            if residual:
                physical += validation[:,:2]
        mse = float(np.mean((physical-val_targets)**2))
        history.append(dict(epoch=epoch,train_standardized_mse=total/len(x),val_mse_c2=mse))
        if mse < best:
            best, stale, best_epoch = mse, 0, epoch
            model.save(path)
        else:
            stale += 1
        if stale >= 30:
            break
    with (output/f'training_hidden{hidden}_seed{seed}.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    restored = ActionMLP.load(path,device=device)
    check = restored.predict_next(validation[:10,:2],validation[:10,2:3],validation[:10,3:])
    assert np.isfinite(check).all()
    return dict(hidden=hidden,seed=seed,best_epoch=best_epoch,val_mse_c2=best,checkpoint=str(path.relative_to(ROOT)))


def main():
    """用验证集比较 64/128 隐层，选定后训练三个固定种子。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--residual',action='store_true')
    parser.add_argument('--rate-weight',type=float,default=0.)
    args = parser.parse_args()
    if not np.isfinite(args.rate_weight) or args.rate_weight < 0:
        parser.error('rate-weight must be finite and nonnegative')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output = ROOT/('reports/week4/results/training_residual' if args.residual else 'reports/week4/results/training')
    if args.rate_weight:
        output = ROOT/f'reports/week4/results/training_rate_{args.rate_weight:g}'
    output.mkdir(parents=True,exist_ok=True)
    features, targets = load_split('train')
    validation, val_targets = load_split('val')
    widths = (64,) if args.residual else (64,128)
    candidates = [train(hidden,42,features,targets,validation,val_targets,device,output,args.residual,args.rate_weight) for hidden in widths]
    selected = min(candidates,key=lambda result:result['val_mse_c2'])
    runs = [selected]+[train(selected['hidden'],seed,features,targets,validation,val_targets,device,output,args.residual,args.rate_weight) for seed in (43,44)]
    config = dict(seeds=[42,43,44],candidates=candidates,selected_hidden=selected['hidden'],
        rate_weight=args.rate_weight,rate_penalty_scale_c=.03,
        runs=runs,optimizer='Adam',learning_rate=.001,batch_size=256,max_epochs=300,patience=30,
        device=device,torch_version=torch.__version__,numpy_version=np.__version__,
        target='delta_state' if args.residual else 'next_state',information='true state + applied action + current observed weather and known internal load',
        input_order=['air','wall','heat_kw','observed_outdoor','load_kw'],
        normalization='training-only population mean and standard deviation; scale floor 1e-6',
        split_manifest=json.loads((ROOT/'data/processed/week4_thermal/manifest.json').read_text()))
    (output/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(dict(device=device,selected_hidden=selected['hidden'],runs=runs),indent=2),flush=True)


if __name__ == '__main__':
    main()
