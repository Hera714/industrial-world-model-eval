"""加载既有 IB 检查点，使用显式历史窗口，GRU 隐状态每次前向从零开始。"""
import numpy as np
import torch
from torch import nn


class IBHistoryDynamics:
    """观测为 IB 原始尺度，动作顺序 velocity/gain/shift，历史由调用方维护。"""
    def __init__(self,path,kind,device='cpu'):
        self.device=torch.device(device)
        self.kind=kind
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        self.network=nn.Module()
        if kind=='ib_gru':
            self.network.encoder=nn.GRU(9,64,batch_first=True)
        elif kind=='ib_history_mlp':
            self.network.encoder=nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(180,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        else:
            self.network.encoder=nn.Sequential(nn.Linear(9,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.network.head=nn.Linear(64,6)
        self.network.load_state_dict(checkpoint['weights'])
        self.network.to(self.device).eval()
        for name,size in (('xmean',9),('xscale',9),('ymean',6),('yscale',6)):
            value=np.asarray(checkpoint[name],dtype=float)
            if value.shape!=(size,) or not np.isfinite(value).all() or ('scale' in name and (value<=0).any()):
                raise ValueError('Invalid checkpoint scaling')
            setattr(self,name,value)
        self.reset()

    def reset(self,*,observation_history=None,action_history=None):
        """保存已发生的19步观测/动作对，空重置清除窗口，不自动复制历史。"""
        self.history=None
        if observation_history is None and action_history is None:
            return
        obs=np.asarray(observation_history,dtype=float)
        actions=np.asarray(action_history,dtype=float)
        if obs.ndim not in (2,3) or obs.shape[-2:]!=(19,6) or actions.shape!=obs.shape[:-1]+(3,):
            raise ValueError('Expected history (...,19,6) and (...,19,3)')
        if not np.isfinite(obs).all() or not np.isfinite(actions).all():
            raise ValueError('Nonfinite history')
        self.history=np.concatenate([obs,actions],axis=-1).copy()

    def predict_window(self,window):
        """输入 (20,9) 或 (N,20,9)，独立批量窗口不共享 GRU 隐状态。"""
        window=np.asarray(window,dtype=float)
        if window.ndim not in (2,3) or window.shape[-2:]!=(20,9) or not np.isfinite(window).all():
            raise ValueError('Expected finite (...,20,9) window')
        single=window.ndim==2
        values=window[None] if single else window
        tensor=torch.tensor((values-self.xmean)/self.xscale,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            if self.kind=='ib_gru':
                encoded=self.network.encoder(tensor)[0][:,-1]
            elif self.kind=='ib_history_mlp':
                encoded=self.network.encoder(tensor)
            else:
                encoded=self.network.encoder(tensor[:,-1])
            delta=self.network.head(encoded).cpu().numpy()*self.yscale+self.ymean
        output=values[:,-1,:6]+delta
        if not np.isfinite(output).all():
            from models.dynamics_base import PredictionOutputError
            raise PredictionOutputError(output[0] if single else output)
        return output[0] if single else output

    def predict_next(self,state,action,exogenous=None):
        """历史模型先 reset 提供历史；此调用不修改历史，候选动作可重复评分。"""
        if exogenous is not None:
            raise ValueError('IB setpoint is already in observation; exogenous must be None')
        state=np.asarray(state,dtype=float)
        if state.ndim not in (1,2) or state.shape[-1]!=6:
            raise ValueError('Invalid IB state shape')
        action=np.broadcast_to(np.asarray(action,dtype=float),state.shape[:-1]+(3,))
        current=np.concatenate([state,action],axis=-1)
        if self.kind=='ib_mlp':
            history=np.repeat(current[...,None,:],19,axis=-2)
        else:
            if self.history is None:
                raise RuntimeError('Initialize history with reset')
            history=np.broadcast_to(self.history,current.shape[:-1]+(19,9))
        return self.predict_window(np.concatenate([history,current[...,None,:]],axis=-2))
