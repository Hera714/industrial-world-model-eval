"""统一模拟环境契约，使用 Gymnasium 的 reset/step 与空间定义。"""
import gymnasium as gym


class BaseEnv(gym.Env):
    """reset 返回 (observation, info)，step 返回五元组。"""
