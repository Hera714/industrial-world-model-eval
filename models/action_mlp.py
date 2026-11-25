"""轻量动作条件 MLP，输入物理单位数组，支持设备迁移和检查点恢复。"""
import numpy as np
import torch
from models.dynamics_base import DynamicsModel


class ActionMLP(DynamicsModel):
    def __init__(self, state_dim=2, action_dim=1, exogenous_dim=2, hidden=128, seed=42, device='cpu', residual=False):
        super().__init__(state_dim, action_dim, exogenous_dim)
        self.config = dict(state_dim=state_dim, action_dim=action_dim, exogenous_dim=exogenous_dim, hidden=hidden, seed=seed)
        self.residual = residual
        self.config['residual'] = residual
        self.device = torch.device(device)
        # 局部设置初始化随机种子，保持调用方 CPU 随机状态。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = torch.nn.Sequential(
                torch.nn.Linear(sum(self.dimensions), hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, state_dim)).to(self.device)
        self.input_mean = np.zeros(sum(self.dimensions))
        self.input_scale = np.ones(sum(self.dimensions))
        self.output_mean = np.zeros(state_dim)
        self.output_scale = np.ones(state_dim)
        self.is_fitted = False

    def _predict(self, state, action, exogenous):
        if not self.is_fitted:
            raise RuntimeError('Train or load a fitted model before inference')
        features = np.concatenate([state, action, exogenous], axis=-1)
        features = (features-self.input_mean)/self.input_scale
        self.network.eval()
        with torch.no_grad():
            prediction = self.network(torch.as_tensor(features, dtype=torch.float32, device=self.device))
        output = prediction.cpu().numpy()*self.output_scale+self.output_mean
        return state+output if self.residual else output

    def save(self, path):
        """标准化参数随网络一起保存，加载时显式选择运行设备。"""
        torch.save(dict(config=self.config, weights=self.network.state_dict(),
            input_mean=self.input_mean.tolist(), input_scale=self.input_scale.tolist(),
            output_mean=self.output_mean.tolist(), output_scale=self.output_scale.tolist(),
            is_fitted=self.is_fitted), path)

    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(**checkpoint['config'], device=device)
        model.network.load_state_dict(checkpoint['weights'])
        for name in ('input_mean','input_scale','output_mean','output_scale'):
            value = np.asarray(checkpoint[name], dtype=float)
            size = sum(model.dimensions) if name.startswith('input') else model.dimensions[0]
            if value.shape != (size,) or not np.isfinite(value).all() or ('scale' in name and (value <= 0).any()):
                raise ValueError('Invalid normalization checkpoint')
            setattr(model, name, value)
        model.is_fitted = checkpoint['is_fitted']
        return model
