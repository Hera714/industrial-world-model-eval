"""预测误差计算；调用者负责先将数值还原为物理单位。"""
import numpy as np


def _validate(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 2 or not actual.size:
        raise ValueError("真实值和预测值必须是形状相同且非空的二维数组")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("真实值或预测值包含 NaN/Inf")
    return actual, predicted


def mae(actual, predicted):
    actual, predicted = _validate(actual, predicted)
    return float(np.abs(predicted - actual).mean())


def rmse(actual, predicted):
    actual, predicted = _validate(actual, predicted)
    return float(np.sqrt(np.square(predicted - actual).mean()))


def error_by_horizon(actual, predicted):
    """沿样本维度计算每个预测步长的 MAE 和 RMSE。"""
    actual, predicted = _validate(actual, predicted)
    error = predicted - actual
    return np.abs(error).mean(axis=0), np.sqrt(np.square(error).mean(axis=0))


def error_by_variable(actual, predicted):
    """二维数组列代表状态变量，返回每列原单位 MAE/RMSE；禁止跨变量混合。"""
    return error_by_horizon(actual, predicted)
