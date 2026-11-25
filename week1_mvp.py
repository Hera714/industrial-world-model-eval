# Week 1 时序预测 MVP：使用 UCI 数据预测未来室内温度 T1。
# 流程：检查数据 → 时间切分及训练集标准化 → 构造窗口 → 训练三个基线
#       → 测试集评测 → 保存逐窗口预测、图表和配置。
# 本实验直接预测未来 24 步，只使用历史观测，不包含控制动作或自由递推。
# 运行：conda activate mvp && python week1_mvp.py

import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 第 1 步：设置项目路径、输出目录、历史长度和预测长度。
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports/week1_results"
OUT.mkdir(parents=True, exist_ok=True)
L, H = 144, 24
FEATURES = ["T1", "T2", "T3", "RH_1", "T_out"]

# 第 2 步：读取并按时间排序，检查 10 分钟采样和缺失值。
df = pd.read_csv(ROOT / "data/raw/energydata_complete.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
if df.date.duplicated().any() or not df.date.diff().dropna().eq(pd.Timedelta(minutes=10)).all():
    raise ValueError("Expected unique timestamps at 10-minute intervals")
if df[FEATURES].isna().any().any():
    raise ValueError("Missing inputs: inspect and handle them before training")

# 第 3 步：按时间划分 70% / 15% / 15%；标准化仅拟合训练集。
n = len(df); a = int(n*.7); b = int(n*.85)
scaler = StandardScaler().fit(df.loc[:a-1, FEATURES])
z = scaler.transform(df[FEATURES]).astype(np.float32)

# 第 4 步：构造滑动窗口。目标必须完全位于对应切分中；允许使用之前已观测到的历史。
def windows(start, end):
    # i 是预测起点；输入截至 i，目标为 i+1 到 i+H。
    origins = np.arange(max(L-1, start-1), end-H)
    # 将历史窗口展平成一维特征；Y 取未来 H 步的 T1。
    X = np.stack([z[i-L+1:i+1].ravel() for i in origins])
    Y = np.stack([z[i+1:i+H+1, 0] for i in origins])
    return X, Y, origins
Xtr,Ytr,itr=windows(L,a); Xv,Yv,iv=windows(a,b); Xt,Yt,it=windows(b,n)
# 确认训练、验证、测试目标时间互不重叠。
assert itr.max()+H < iv.min()+1 and iv.max()+H < it.min()+1

# 第 5 步：准备反标准化，将温度恢复为摄氏度后计算指标。
mean, scale = scaler.mean_[0], scaler.scale_[0]
def original(y): return y*scale+mean
truth=original(Yt)

# 第 6 步：Persistence 保持最后观测值；Ridge 使用历史特征直接预测多步。
pred={"persistence": np.repeat(Xt[:, -len(FEATURES):][:,:1],H,axis=1)}
linear=Ridge(alpha=10.0).fit(Xtr,Ytr)
pred["linear"]=linear.predict(Xt)

# 第 7 步：训练 MLP。三个固定种子用于观察训练结果的波动。
device = "cuda" if torch.cuda.is_available() else "cpu"
for seed in (42,43,44):
    # 固定随机种子，并优先使用可用 GPU。
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    # 两层隐藏层，每层 64 个神经元；输出 H 个未来温度。
    model=nn.Sequential(nn.Linear(Xtr.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,H)).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Ytr)),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float("inf"); state=None; patience=0
    # 每轮先训练，再用验证集评估；连续 5 轮没有改善则提前停止。
    for epoch in range(40):
        model.train()
        for x,y in loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=((model(x)-y)**2).mean(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val=((model(torch.from_numpy(Xv).to(device))-torch.from_numpy(Yv).to(device))**2).mean().item()
        # 保存验证误差最小时的模型参数，不用测试集选模型。
        if val < best-1e-5:
            best=val; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else:
            patience+=1
            if patience>=5: break
    # 恢复最佳参数，在测试集上预测并保存权重。
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): pred[f"mlp_{seed}"]=model(torch.from_numpy(Xt).to(device)).cpu().numpy()
    torch.save(state,OUT / f"mlp_{seed}.pt")
    print(f"seed={seed} epochs={epoch+1} val_mse={best:.5f}",flush=True)

# 第 8 步：逐预测步长计算 MAE、RMSE，并保存每个窗口的真实值和预测值。
rows=[]; traces=[]
for name, val in pred.items():
    yhat=original(val); err=yhat-truth
    for h in range(H): rows.append(dict(model=name,horizon=h+1,mae=float(np.abs(err[:,h]).mean()),rmse=float(np.sqrt((err[:,h]**2).mean()))))
    for j,origin in enumerate(it):
        for h in range(H): traces.append((name,str(df.date.iloc[origin]),str(df.date.iloc[origin+h+1]),h+1,float(truth[j,h]),float(yhat[j,h])))
metrics=pd.DataFrame(rows); metrics.to_csv(OUT/"metrics.csv",index=False)
pd.DataFrame(traces,columns=["model","origin","target_time","horizon","actual_c","predicted_c"]).to_csv(OUT/"predictions.csv",index=False)

# 第 9 步：绘制各模型误差随预测步长变化的曲线。
fig,ax=plt.subplots()
for name,g in metrics.groupby("model"): ax.plot(g.horizon,g.mae,label=name)
ax.set(xlabel="Horizon (10-minute steps)",ylabel="MAE (°C)"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"error_by_horizon.png",dpi=150); plt.close(fig)

# 第 10 步：绘制第一个测试窗口的真实温度与预测轨迹。
fig,ax=plt.subplots(); ax.plot(range(1,H+1),truth[0],label="actual",linewidth=2)
for name,val in pred.items(): ax.plot(range(1,H+1),original(val)[0],label=name)
ax.set(xlabel="Horizon (10-minute steps)",ylabel="T1 (°C)"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"forecast_example.png",dpi=150); plt.close(fig)

# 第 11 步：保存切分边界、窗口数、标准化参数、种子和运行环境。
meta=dict(seed=[42,43,44],history=L,horizon=H,features=FEATURES,train_end=str(df.date.iloc[a-1]),val_start=str(df.date.iloc[a]),val_end=str(df.date.iloc[b-1]),test_start=str(df.date.iloc[b]),test_end=str(df.date.iloc[-1]),train_windows=len(Xtr),val_windows=len(Xv),test_windows=len(Xt),mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),device=device,torch=torch.__version__)
(OUT/"config.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
# 在终端显示 1、6、12、24 步的测试指标。
print(metrics[metrics.horizon.isin([1,6,12,24])].to_string(index=False))
