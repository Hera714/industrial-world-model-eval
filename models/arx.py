"""历史状态与外部观测的多输出 Ridge 预测，不含控制动作。"""
import numpy as np
from sklearn.linear_model import Ridge
from .base import ForecastModel

class LinearAutoregression(ForecastModel):
    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0)):
        self.alphas = alphas

    def fit(self, X, y, X_val=None, y_val=None):
        X = self.check_X(X).astype(np.float64); X_val = self.check_X(X_val).astype(np.float64)
        self.validation_mse = {}
        for alpha in self.alphas:
            candidate = Ridge(alpha=alpha).fit(X.reshape(len(X), -1), y)
            err = candidate.predict(X_val.reshape(len(X_val), -1)) - y_val
            self.validation_mse[str(alpha)] = float(np.mean(err ** 2))
        self.alpha = min(self.alphas, key=lambda a: self.validation_mse[str(a)])
        self.model = Ridge(alpha=self.alpha).fit(X.reshape(len(X), -1), y)
        return self

    def predict(self, X):
        X = self.check_X(X)
        return self.model.predict(X.reshape(len(X), -1))
