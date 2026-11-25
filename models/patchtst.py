"""小规模单变量 PatchTST：共享 patch embedding、Transformer、直接多步预测。"""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .base import ForecastModel

class PatchTST(ForecastModel):
    def __init__(self, history=144, horizon=24, patch=16, stride=8, d_model=32, heads=4, layers=2, seed=42, epochs=30, patience=5, batch=128, device=None):
        self.history,self.horizon,self.patch,self.stride=history,horizon,patch,stride
        self.d_model,self.heads,self.layers,self.seed,self.epochs,self.patience,self.batch=d_model,heads,layers,seed,epochs,patience,batch
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    def fit(self,X,y,X_val=None,y_val=None):
        X=self.check_X(X); X_val=self.check_X(X_val)
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.seed)
        n_patches=(self.history-self.patch)//self.stride+1
        class Net(nn.Module):
            def __init__(net):
                super().__init__(); net.embed=nn.Linear(self.patch,self.d_model); net.pos=nn.Parameter(torch.zeros(1,n_patches,self.d_model))
                layer=nn.TransformerEncoderLayer(self.d_model,self.heads,dim_feedforward=128,dropout=.1,batch_first=True,activation='gelu')
                net.encoder=nn.TransformerEncoder(layer,self.layers,enable_nested_tensor=False); net.head=nn.Linear(n_patches*self.d_model,self.horizon)
            def forward(net,x):
                # 只使用 T1 历史；每个 patch 作为一个 token。
                patches=x[:,:,0].unfold(1,self.patch,self.stride)
                return net.head(net.encoder(net.embed(patches)+net.pos).flatten(1))
        self.model=Net().to(self.device); opt=torch.optim.Adam(self.model.parameters(),lr=.001)
        loader=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(np.asarray(y,dtype=np.float32))),batch_size=self.batch,shuffle=True,generator=torch.Generator().manual_seed(self.seed))
        xv=torch.from_numpy(X_val).to(self.device); yv=torch.from_numpy(np.asarray(y_val,dtype=np.float32)).to(self.device)
        best=float('inf'); remaining=self.patience
        for epoch in range(self.epochs):
            self.model.train()
            for xb,yb in loader:
                xb,yb=xb.to(self.device),yb.to(self.device); opt.zero_grad(); loss=((self.model(xb)-yb)**2).mean(); loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad(): score=((self.model(xv)-yv)**2).mean().item()
            print(f'epoch={epoch+1} val_mse={score:.5f}',flush=True)
            if score < best-1e-5:
                best=score; self.best_epoch=epoch+1; remaining=self.patience; state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
            else:
                remaining-=1
                if not remaining: break
        self.validation_mse=best; self.model.load_state_dict(state); return self
    def predict(self,X):
        X=self.check_X(X); self.model.eval(); outputs=[]
        with torch.no_grad():
            for chunk in np.array_split(X,max(1,(len(X)+511)//512)):
                outputs.append(self.model(torch.from_numpy(chunk).to(self.device)).cpu().numpy())
        return np.concatenate(outputs)
