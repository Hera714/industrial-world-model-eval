"""统一直接多步预测接口；输入为 (样本, 历史步数, 特征数)。"""
from abc import ABC, abstractmethod
import numpy as np

class ForecastModel(ABC):
    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None):
        """只用训练集拟合，验证集可用于选择超参数。"""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """返回形状为 (样本, 预测步数) 的标准化预测。"""

    @staticmethod
    def check_X(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or not np.isfinite(X).all():
            raise ValueError("X 必须是有限数值的三维数组")
        return X
