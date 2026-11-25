"""保持最后一个目标观测值的直接多步基线。"""
import numpy as np
from .base import ForecastModel

class Persistence(ForecastModel):
    def __init__(self, horizon=24, target_index=0):
        self.horizon, self.target_index = horizon, target_index

    def fit(self, X, y, X_val=None, y_val=None):
        self.check_X(X)
        return self

    def predict(self, X):
        X = self.check_X(X)
        return np.repeat(X[:, -1, self.target_index, None], self.horizon, axis=1)
