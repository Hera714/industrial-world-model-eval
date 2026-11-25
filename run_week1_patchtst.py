"""P1 PatchTST 与相同 T1 历史信息集的线性及保持基线对比。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models.patchtst import PatchTST
from models.persistence import Persistence
from models.arx import LinearAutoregression
from metrics.prediction import mae,rmse,error_by_horizon

def load(name):
    with np.load(f'data/processed/week1/{name}.npz') as f: return {k:f[k] for k in f.files}

def main():
    out=Path('reports/week1_p1'); out.mkdir(parents=True,exist_ok=True)
    meta=json.loads(Path('data/processed/week1/metadata.json').read_text()); tr=load('train'); val=load('val')
    idx=meta['features'].index('T1'); xtr=tr['X'][:,:,idx:idx+1]; xv=val['X'][:,:,idx:idx+1]
    models={'persistence_t1':Persistence(meta['horizon']), 'linear_t1':LinearAutoregression(), 'patchtst_t1':PatchTST(history=meta['history'],horizon=meta['horizon'])}
    for name,model in models.items():
        print('训练',name,flush=True); model.fit(xtr,tr['y'],xv,val['y'])
    test=load('test'); xt=test['X'][:,:,idx:idx+1]; mu=meta['scaler']['mean']['T1']; scale=meta['scaler']['scale']['T1']; actual=test['y']*scale+mu
    results=[]; predictions={'actual':actual,'target_start':test['target_start']}
    for name,model in models.items():
        p=model.predict(xt)*scale+mu
        if p.shape!=actual.shape or not np.isfinite(p).all(): raise ValueError('预测形状或数值不合法')
        predictions[name]=p; a,r=error_by_horizon(actual,p)
        results.append({'model':name,'horizon':'all','mae_c':mae(actual,p),'rmse_c':rmse(actual,p)})
        for h in (1,6,12,24): results.append({'model':name,'horizon':h,'mae_c':a[h-1],'rmse_c':r[h-1]})
    np.savez_compressed(out/'patchtst_predictions.npz',**predictions)
    pd.DataFrame(results).to_csv(out/'patchtst_metrics.csv',index=False)
    m=models['patchtst_t1']; torch.save(m.model.state_dict(),out/'patchtst.pt')
    cfg={'information_set':'T1 history only','history':m.history,'horizon':m.horizon,'patch_length':m.patch,'stride':m.stride,'d_model':m.d_model,'heads':m.heads,'layers':m.layers,'seed':m.seed,'max_epochs':m.epochs,'patience':m.patience,'batch_size':m.batch,'learning_rate':.001,'best_epoch':m.best_epoch,'validation_mse':m.validation_mse,'device':m.device,'test_windows':len(actual),'linear_selected_alpha':models['linear_t1'].alpha,'torch_version':torch.__version__}
    (out/'patchtst_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8'); print(pd.DataFrame(results).to_string(index=False))

if __name__=='__main__': main()
