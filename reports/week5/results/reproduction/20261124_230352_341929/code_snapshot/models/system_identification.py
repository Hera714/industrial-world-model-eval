# 总说明：仅用训练回合拟合线性动作条件动力学，支持批量一步预测。
import numpy as np
from models.dynamics_base import DynamicsModel

class LinearDynamics(DynamicsModel):
    def __init__(self, exogenous_dim=1):
        # 默认一维室外温度，兼容 Week 2；Week 4 显式设为 2。
        super().__init__(2, 1, exogenous_dim)
    def fit(self, state, action, outdoor, next_state):
        x,u,d=self.inputs(state,action,outdoor); y=np.asarray(next_state,float)
        if x.ndim != 2:
            raise ValueError('Fit requires a batch')
        design=np.column_stack([x,u,d,np.ones(len(x))])
        if x.ndim!=2 or x.shape[1]!=2 or y.shape!=x.shape or not np.isfinite(design).all() or not np.isfinite(y).all(): raise ValueError('训练数据无效')
        self.coef=np.linalg.lstsq(design,y,rcond=None)[0]
        return self
    def _predict(self,x,u,d):
        if not hasattr(self, 'coef'):
            raise RuntimeError('Fit model before inference')
        return np.concatenate([x,u,d,np.ones(x.shape[:-1]+(1,))],axis=-1)@self.coef
    def save(self,path):
        np.savez(path, coef=self.coef, exogenous_dim=self.dimensions[2])
    @classmethod
    def load(cls,path):
        with np.load(path, allow_pickle=False) as saved:
            model=cls(exogenous_dim=int(saved['exogenous_dim']))
            model.coef=saved['coef']
        if model.coef.shape != (sum(model.dimensions)+1,2) or not np.isfinite(model.coef).all():
            raise ValueError('Invalid linear checkpoint')
        return model
    def parameters(self):
        return {'A':self.coef[:2].T.tolist(),'B':self.coef[2:3].T.tolist(),'E':self.coef[3:-1].T.tolist(),'b':self.coef[-1].tolist()}
