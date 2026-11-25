"""官方 PatchTST 自监督预训练 + T1 监督微调；独立通道共享编码器。"""
import json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from metrics.prediction import mae, rmse, error_by_horizon

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'external/PatchTST/PatchTST_self_supervised'))
from src.models.patchTST import PatchTST
from src.callback.patch_mask import create_patch, random_masking

OUT=ROOT/'reports/week1_p1/official_patchtst'

def load(name):
    with np.load(ROOT/f'data/processed/week1/{name}.npz',allow_pickle=False) as f: return {k:f[k] for k in f.files}

def main():
    OUT.mkdir(parents=True,exist_ok=True); seed=42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    meta=json.loads((ROOT/'data/processed/week1/metadata.json').read_text()); tr=load('train'); val=load('val')
    # 非重叠 patch：16 步长与 patch 长度一致；40% 随机遮盖。
    patch=16; stride=16; n_patch=(meta['history']-patch)//stride+1; mask_ratio=.4
    args=dict(c_in=5,target_dim=24,patch_len=patch,stride=stride,num_patch=n_patch,n_layers=2,n_heads=4,d_model=32,d_ff=128,shared_embedding=True,dropout=.1,head_dropout=.1,act='relu',res_attention=False)
    pre=PatchTST(**args,head_type='pretrain').to(device); opt=torch.optim.Adam(pre.parameters(),lr=.001)
    loader=DataLoader(TensorDataset(torch.from_numpy(tr['X'])),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
    xv=torch.from_numpy(val['X']).to(device)
    def masked_loss(net,x):
        patches,_=create_patch(x,patch,stride); masked,_,mask,_=random_masking(patches,mask_ratio)
        result=net(masked); return (((result-patches)**2).mean(dim=-1)*mask).sum()/mask.sum()
    best=float('inf'); prelog=[]
    for epoch in range(6):
        pre.train(); losses=[]
        for (xb,) in loader:
            opt.zero_grad(); loss=masked_loss(pre,xb.to(device)); loss.backward(); opt.step(); losses.append(loss.item())
        pre.eval(); torch.manual_seed(100)
        with torch.no_grad(): score=masked_loss(pre,xv).item()
        prelog.append({'epoch':epoch+1,'train_masked_mse':float(np.mean(losses)),'validation_masked_mse':score})
        print('pretrain',epoch+1,score,flush=True)
        if score<best: best=score; state={k:v.detach().cpu().clone() for k,v in pre.state_dict().items()}; pre_epoch=epoch+1
    pre.load_state_dict(state); torch.save(state,OUT/'pretrain.pt')
    model=PatchTST(**args,head_type='prediction').to(device); model.backbone.load_state_dict(pre.backbone.state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=.0005); best=float('inf'); patience=0; finelog=[]
    loader=DataLoader(TensorDataset(torch.from_numpy(tr['X']),torch.from_numpy(tr['y'])),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
    yv=torch.from_numpy(val['y']).to(device)
    for epoch in range(15):
        model.train(); losses=[]
        for xb,yb in loader:
            opt.zero_grad(); patches,_=create_patch(xb.to(device),patch,stride); pred=model(patches)[:,:,0]; loss=((pred-yb.to(device))**2).mean(); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            scores=[]
            for start in range(0,len(xv),256):
                p,_=create_patch(xv[start:start+256],patch,stride); scores.append(((model(p)[:,:,0]-yv[start:start+256])**2).sum().item())
            score=sum(scores)/yv.numel()
        finelog.append({'epoch':epoch+1,'train_mse':float(np.mean(losses)),'validation_mse':score}); print('finetune',epoch+1,score,flush=True)
        if score<best-1e-5: best=score; best_epoch=epoch+1; patience=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience+=1
            if patience>=4: break
    model.load_state_dict(state); model.eval(); torch.save(state,OUT/'finetuned.pt')
    test=load('test'); xt=torch.from_numpy(test['X']); preds=[]
    with torch.no_grad():
        for start in range(0,len(xt),256):
            p,_=create_patch(xt[start:start+256].to(device),patch,stride); preds.append(model(p)[:,:,0].cpu().numpy())
    mu=meta['scaler']['mean']['T1']; sd=meta['scaler']['scale']['T1']; pred=np.concatenate(preds)*sd+mu; actual=test['y']*sd+mu
    a,r=error_by_horizon(actual,pred); rows=[{'horizon':int(h),'mae_c':float(a[h-1]),'rmse_c':float(r[h-1])} for h in range(1,25)]
    pd.DataFrame(rows).to_csv(OUT/'metrics_by_horizon.csv',index=False); pd.DataFrame(prelog).to_csv(OUT/'pretrain_log.csv',index=False); pd.DataFrame(finelog).to_csv(OUT/'finetune_log.csv',index=False)
    np.savez_compressed(OUT/'test_predictions.npz',actual=actual,predicted=pred,target_start=test['target_start'])
    cfg={'official_commit':'204c21efe0b39603ad6e2ca640ef5896646ab1a9','features':meta['features'],'target':'T1','channel_independence':True,'pretrain_mask_ratio':mask_ratio,'patch_length':patch,'stride':stride,'model':args,'seed':seed,'pretrain_best_epoch':pre_epoch,'finetune_best_epoch':best_epoch,'pretrain_epochs':6,'finetune_max_epochs':15,'device':device,'overall_mae_c':mae(actual,pred),'overall_rmse_c':rmse(actual,pred),'note':'Only T1 forecast loss; channel-independent backbone does not mix other feature values into T1 prediction.'}
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8'); print(json.dumps(cfg,indent=2))

if __name__=='__main__': main()
