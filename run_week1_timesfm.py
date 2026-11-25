"""TimesFM 3.0 零样本：同一批 UCI 测试窗口、T1 历史和 24 步预测。"""
import json, os, time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
import timesfm
from metrics.prediction import mae, rmse, error_by_horizon

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data/processed/week1'; OUT=ROOT/'reports/week1/week1_p2/timesfm'

def load_npz(path):
    with np.load(path,allow_pickle=False) as f: return {k:f[k] for k in f.files}

def main():
    OUT.mkdir(parents=True,exist_ok=True); os.environ['HF_HUB_OFFLINE']='1'
    meta=json.loads((DATA/'metadata.json').read_text()); test=load_npz(DATA/'test.npz')
    mu=meta['scaler']['mean']['T1']; scale=meta['scaler']['scale']['T1']; idx=meta['features'].index('T1'); h=meta['horizon']
    # 还原摄氏度；模型不使用其他变量或未来协变量。
    histories=(test['X'][:,:,idx]*scale+mu).astype(np.float32); truth=test['y']*scale+mu
    device='cuda' if torch.cuda.is_available() else 'cpu'; started=time.perf_counter()
    forecaster=timesfm.TimesFM3Forecaster.from_pretrained('google/timesfm-3.0-pytorch',device=device)
    load_seconds=time.perf_counter()-started; preds=[]; batch=32
    if device=='cuda': torch.cuda.synchronize()
    started=time.perf_counter()
    for start in range(0,len(histories),batch):
        outputs=list(forecaster.predict_batch([x for x in histories[start:start+batch]],horizon=h,return_quantiles=False,use_symmetric_averaging=False,use_znorm=False))
        preds.extend([np.asarray(o.forecast).reshape(-1) for o in outputs])
        if start%640==0: print(f'预测 {min(start+batch,len(histories))}/{len(histories)}',flush=True)
    if device=='cuda': torch.cuda.synchronize()
    seconds=time.perf_counter()-started; pred=np.stack(preds)
    if pred.shape!=truth.shape or not np.isfinite(pred).all(): raise ValueError(f'预测形状或数值异常：{pred.shape}')
    a,r=error_by_horizon(truth,pred); table=pd.DataFrame({'model':'timesfm3_t1','horizon':np.arange(1,h+1),'mae_c':a,'rmse_c':r})
    table.to_csv(OUT/'metrics_by_horizon.csv',index=False); np.savez_compressed(OUT/'predictions.npz',actual=truth,predicted=pred,target_start=test['target_start'])
    # 严格核对已有预测与测试目标、时间戳一致，再合并结果。
    comparisons=[]
    def add(name,p,information):
        comparisons.append({'model':name,'information':information,'mae_c':mae(truth,p),'rmse_c':rmse(truth,p)})
    add('timesfm3_t1',pred,'T1 history only')
    for path,labels in [(ROOT/'reports/week1/week1_baselines/test_predictions.npz',{'persistence':'T1 history only','linear':'five historical features','mlp_42':'five historical features','mlp_43':'five historical features','mlp_44':'five historical features'}),(ROOT/'reports/week1/week1_p1/patchtst_predictions.npz',{'linear_t1':'T1 history only','patchtst_t1':'T1 history only; simplified old model'}),(ROOT/'reports/week1/week1_p1/official_patchtst/test_predictions.npz',{'predicted':'five-channel shared pretraining; T1 independent inference'})]:
        data=load_npz(path)
        if not np.array_equal(data['target_start'],test['target_start']) or not np.allclose(data['actual'],truth,atol=1e-5): raise ValueError(f'窗口或目标不一致: {path}')
        for key,info in labels.items(): add('official_patchtst_t1' if key=='predicted' else key,data[key],info)
    pd.DataFrame(comparisons).to_csv(OUT/'model_comparison.csv',index=False)
    revision_path=Path.home()/'.cache/huggingface/hub/models--google--timesfm-3.0-pytorch/refs/main'
    config={'model':'google/timesfm-3.0-pytorch','cached_revision':revision_path.read_text().strip() if revision_path.exists() else None,'timesfm_package':version('timesfm'),'torch':torch.__version__,'device':device,'history':meta['history'],'horizon':h,'test_windows':len(histories),'batch_size':batch,'inputs':'T1 history in degrees Celsius','covariates':'none','use_znorm':False,'use_symmetric_averaging':False,'load_seconds':load_seconds,'inference_seconds':seconds,'timing_scope':'entire predict_batch loop including first-call warmup and output conversion; CUDA synchronized','overall_mae_c':mae(truth,pred),'overall_rmse_c':rmse(truth,pred)}
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(config,indent=2)); print(pd.DataFrame(comparisons).to_string(index=False)); print(table[table.horizon.isin([1,6,12,24])].to_string(index=False))

if __name__=='__main__': main()
