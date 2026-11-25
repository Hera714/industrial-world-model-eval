"""总说明：通过统一配置加载 BOPTEST 五类动力学模型，外部统一使用物理单位 NumPy 数组。"""
from pathlib import Path
import numpy as np
import torch
from benchmark.config import ConfigurationError, load_config, validate_config
from models.dynamics_base import load_boptest_dynamics


def check_checkpoint_dimensions(data, history_steps, path):
    """检查标准化向量及线性系数；网络参数形状由原模型的严格加载继续检查。"""
    for field, size in [('mean', 12), ('scale', 12), ('ymean', 6), ('yscale', 6)]:
        values = np.asarray(data.get(field), dtype=float)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ConfigurationError(f'{path}.{field}: expected finite vector ({size},)')
        if field in ('scale', 'yscale') and np.any(values <= 0):
            raise ConfigurationError(f'{path}.{field}: normalization scale must be positive')
    if data.get('kind') == 'arx':
        coef = np.asarray(data.get('coef'), dtype=float)
        if coef.shape != (history_steps * 12 + 1, 6) or not np.isfinite(coef).all():
            raise ConfigurationError(f'{path}.coef: incompatible input/output dimensions')


class LoadedBoptestModel:
    """显式历史接口：单窗口 (16,12) 或批量 (N,16,12)，输出对应 (6,) 或 (N,6)。"""
    def __init__(self, model, name, device):
        self.model, self.name = model, name
        self.kind, self.length = model.kind, model.length
        self.device = str(device)
        self.input_features, self.output_features = 12, 6
        if hasattr(model, 'network'):
            model.network.to(device).eval()

    def reset(self):
        """历史由调用者逐窗口传入；GRU 每次 forward 使用零隐藏状态，无跨窗口记忆。"""
        pass

    def predict(self, history):
        """保留摄氏度/kW/kWh；仅神经网络的标准化张量进入配置指定设备。"""
        values = np.asarray(history, dtype=np.float64)
        single = values.ndim == 2
        if single:
            values = values[None]
        if (values.ndim != 3 or values.shape[1:] != (self.length, 12)
                or len(values) == 0 or not np.isfinite(values).all()):
            raise ValueError(f'{self.name}: expected finite nonempty history (N,{self.length},12)')
        model = self.model
        if hasattr(model, 'network'):
            normalized = (values - model.mean) / model.scale
            with torch.inference_mode():
                tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
                output = model.network(tensor).detach().cpu().numpy()
            output = output * model.yscale + model.ymean
            if self.kind == 'residual_mlp':
                output[..., :5] += values[:, -1, :5]
        else:
            output = model.predict(values)
        output = np.asarray(output, dtype=np.float64)
        if output.shape != (len(values), 6) or not np.isfinite(output).all():
            from models.dynamics_base import PredictionOutputError
            raise PredictionOutputError(output)
        return output[0] if single else output

    def predict_transition(self, history):
        """第六个输出是区间能耗，不是需要递推的观测状态。"""
        output = self.predict(history)
        return {'next_observation': output[..., :5], 'interval_energy_kwh': output[..., 5]}


def load_models(config, root=None):
    """先完成全部本地预检，再加载模型；不创建模拟器或访问服务。返回按配置名称索引的字典。"""
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    cfg = load_config(config, root) if isinstance(config, (str, Path)) else validate_config(config, root)
    device = torch.device(cfg['model']['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise ConfigurationError('model.device: requested CUDA device is unavailable')
    result = {}
    for spec in cfg['model']['instances']:
        name = spec['name']
        try:
            # 复用原有加载入口的哈希、模型类型和网络严格形状检查。
            model = load_boptest_dynamics(dict(kind=spec['kind'], checkpoint=spec['checkpoint']['path'],
                                              sha256=spec['checkpoint']['sha256']), root)
            wrapped = LoadedBoptestModel(model, name, device)
            # 均值只用于无仿真的接口探测，不作为实际评测样本。
            history = np.tile(model.mean, (model.length, 1))
            wrapped.predict(history)
            wrapped.predict(np.stack([history, history]))
            result[name] = wrapped
        except Exception as error:
            raise ConfigurationError(f'model.instances.{name}: loading failed: {error}') from error
    return result
