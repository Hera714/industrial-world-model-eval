"""动作条件动力学统一契约，外部使用物理单位 NumPy，模型内部处理设备。"""
from abc import ABC, abstractmethod
import numpy as np


class PredictionOutputError(ValueError):
    """携带未修正的非法模型输出，允许评测区分 NaN/Inf 与推理异常。"""
    def __init__(self, prediction):
        super().__init__('Invalid predicted state')
        self.prediction = np.asarray(prediction).copy()


class DynamicsModel(ABC):
    """单状态 (S,) 或批量 (N,S)，输出始终与状态形状一致。"""
    def __init__(self, state_dim=2, action_dim=1, exogenous_dim=2):
        self.dimensions = (state_dim, action_dim, exogenous_dim)
        if any(type(dim) is not int or dim < 1 for dim in self.dimensions):
            raise ValueError('Dimensions must be positive integers')

    def inputs(self, state, action, exogenous):
        """一维动作批量、标量广播兼容已有 MPC；其余维度必须明确匹配。"""
        x = np.asarray(state, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != self.dimensions[0] or x.size == 0:
            raise ValueError('Invalid state shape')
        values = [x]
        for value, dim in zip((action, exogenous), self.dimensions[1:]):
            array = np.asarray(value, dtype=np.float64)
            target = x.shape[:-1] + (dim,)
            if dim == 1 and array.shape == x.shape[:-1]:
                array = array[..., None]
            try:
                array = np.broadcast_to(array, target)
            except ValueError as error:
                raise ValueError('Invalid action or exogenous shape') from error
            values.append(array)
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError('Inputs must be finite')
        return values

    def predict_next(self, state, action, exogenous):
        values = self.inputs(state, action, exogenous)
        output = np.asarray(self._predict(*values), dtype=np.float64)
        if output.shape != values[0].shape or not np.isfinite(output).all():
            raise PredictionOutputError(output)
        return output

    def reset(self, *, observation_history=None, action_history=None):
        """无记忆模型接受重置，历史只供需要记忆的模型使用。"""
        if observation_history is not None or action_history is not None:
            raise ValueError('This model does not consume history')


    @abstractmethod
    def _predict(self, state, action, exogenous):
        pass


def load_dynamics(spec, root='.', device='cpu', hashes=None):
    """按配置加载模型并校验可选 SHA-256，外部始终使用物理单位数组。"""
    from pathlib import Path
    import hashlib
    path = Path(root)/spec['checkpoint'] if spec.get('checkpoint') else None
    expected = spec.get('sha256') or (hashes or {}).get(spec.get('checkpoint'))
    if expected and path is not None:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Checkpoint SHA256 mismatch')
    kind=spec['kind']
    if kind == 'rate_projection':
        from models.physics_constraints import RateConstrainedDynamics
        model = RateConstrainedDynamics(load_dynamics(spec['base_model'], root, device, hashes),
            outdoor_bounds=tuple(spec['outdoor_bounds_c']), noise_margin=spec['noise_margin_c'])
    elif kind == 'persistence':
        model=PersistenceDynamics(**spec.get('dimensions',{}))
    elif kind == 'linear':
        from models.system_identification import LinearDynamics
        model=LinearDynamics.load(path)
    elif kind == 'action_mlp':
        from models.action_mlp import ActionMLP
        model=ActionMLP.load(path,device=device)
    elif kind in ('ib_mlp','ib_history_mlp','ib_gru'):
        from models.history_dynamics import IBHistoryDynamics
        model=IBHistoryDynamics(path,kind,device)
    else:
        raise ValueError(f'Unknown dynamics kind: {kind}')
    return model

class PersistenceDynamics(DynamicsModel):
    """接受统一动作和扰动输入，预测保持当前完整状态。"""
    def _predict(self, state, action, exogenous):
        return state.copy()

    def save(self, path):
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.dimensions))

    @classmethod
    def load(cls, path):
        import json
        from pathlib import Path
        return cls(*json.loads(Path(path).read_text()))


def load_boptest_dynamics(spec, root='.'):
    """加载BOPTEST历史观测/区间能耗模型，校验检查点SHA256并统一单样本和批量预测接口。"""
    from pathlib import Path
    import hashlib
    from models.boptest_dynamics import BoptestHistoryModel
    path = Path(root) / spec['checkpoint']
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('BOPTEST checkpoint SHA256 mismatch')
    model = BoptestHistoryModel.load(path)
    if model.kind != spec['kind']:
        raise ValueError('BOPTEST model kind mismatch')
    return model
