"""总说明：通用观测历史 ARX/MLP/GRU，预测下一公开观测及本控制步能耗；无隐藏状态跨窗口传递。"""
from pathlib import Path
import numpy as np
import torch
from torch import nn


class HistoryNetwork(nn.Module):
    """所有模型共享相同历史字段；GRU 每次 forward 从零隐藏状态独立编码窗口。"""
    def __init__(self,kind,length,features=12,outputs=6):
        super().__init__();self.kind=kind
        if kind=='gru':
            self.encoder=nn.GRU(features,64,batch_first=True);self.head=nn.Linear(64,outputs)
        else:
            self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(length*features,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
            self.head=nn.Linear(64,outputs)
    def forward(self,value):
        if self.kind=='gru': value=self.encoder(value)[0][:,-1]
        else: value=self.encoder(value)
        return self.head(value)


class BoptestHistoryModel:
    """外部物理单位 history (N,L,12)，输出 (N,6)：5维观测与区间能耗kWh。"""
    def __init__(self,kind,length=16,seed=42):
        self.kind=kind;self.length=length;self.seed=seed
        self.mean=np.zeros(12);self.scale=np.ones(12);self.ymean=np.zeros(6);self.yscale=np.ones(6)
        if kind in ('mlp','gru','residual_mlp'):
            with torch.random.fork_rng():
                torch.manual_seed(seed);self.network=HistoryNetwork(kind,length)
    def predict(self,history):
        values=np.asarray(history,float);single=values.ndim==2
        if single: values=values[None]
        if values.shape[1:]!=(self.length,12) or not np.isfinite(values).all(): raise ValueError('Invalid BOPTEST history')
        if self.kind=='persistence': result=np.column_stack([values[:,-1,:5],values[:,-1,4]*.25])
        else:
            x=(values-self.mean)/self.scale
            if self.kind=='arx': result=np.column_stack([x.reshape(len(x),-1),np.ones(len(x))])@self.coef
            else:
                self.network.eval()
                with torch.no_grad(): result=self.network(torch.as_tensor(x,dtype=torch.float32)).numpy()
            result=result*self.yscale+self.ymean
            if self.kind == 'residual_mlp':
                result[...,:5] += values[:,-1,:5]
        if result.shape!=(len(values),6) or not np.isfinite(result).all(): raise ValueError('Invalid BOPTEST prediction')
        return result[0] if single else result
    def save(self,path):
        data=dict(kind=self.kind,length=self.length,seed=self.seed,mean=self.mean.tolist(),scale=self.scale.tolist(),
            ymean=self.ymean.tolist(),yscale=self.yscale.tolist())
        if self.kind=='arx':data['coef']=self.coef.tolist()
        elif self.kind!='persistence':data['weights']=self.network.state_dict()
        torch.save(data,path)
    def predict_transition(self,history):
        """显式区分下一观测与本步能耗，避免将区间能耗误认为要递推的建筑状态。"""
        output=self.predict(history)
        return {'next_observation':output[...,:5], 'interval_energy_kwh':output[...,5]}
    @classmethod
    def load(cls,path):
        data=torch.load(path,map_location='cpu',weights_only=True);model=cls(data['kind'],data['length'],data['seed'])
        for field in ('mean','scale','ymean','yscale'):setattr(model,field,np.asarray(data[field]))
        if model.kind=='arx':model.coef=np.asarray(data['coef'])
        elif model.kind!='persistence':model.network.load_state_dict(data['weights'])
        return model


def exogenous_features(weather,timestamp):
    """天气为 °C/Wm-2，日/年周期是未来已知日历；绝不使用未来室温作为外生输入。"""
    t=np.asarray(timestamp);weather=np.asarray(weather)
    return np.concatenate([weather,np.stack([np.sin(t*2*np.pi/86400),np.cos(t*2*np.pi/86400),
        np.sin(t*2*np.pi/(365*86400)),np.cos(t*2*np.pi/(365*86400))],axis=-1)],axis=-1)
