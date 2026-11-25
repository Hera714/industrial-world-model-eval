"""用 Gymnasium 接口运行 BOPTEST 案例内置控制器。"""
import json
from urllib.request import Request, urlopen

import numpy as np
from gymnasium import spaces
from simulators.base_env import BaseEnv


class BoptestEnv(BaseEnv):
    """唯一动作 0 表示保持案例内置控制，不发送覆盖命令。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(config["measurements"]),), dtype=np.float64)
        self.testid = None
        self.done = True

    def request(self, method, path, data=None):
        """检查 HTTP 状态及服务响应状态。推进请求失败时由调用方结束回合。"""
        request = Request(self.config["url"].rstrip("/") + path,
                          None if data is None else json.dumps(data).encode(),
                          {"Content-Type": "application/json"}, method=method)
        with urlopen(request, timeout=120) as response:
            # 本地服务的 stop 成功响应是文本，不是 JSON。
            if method == "PUT" and path.startswith("/stop/"):
                return None
            result = json.load(response)
        if result.get("status", 200) != 200:
            raise RuntimeError(result.get("message", path))
        return result.get("payload", result)

    def _sample(self, points):
        obs = np.array([points[name] for name in self.config["measurements"]], dtype=np.float64)
        if not np.isfinite(obs).all():
            raise RuntimeError("BOPTEST returned nonfinite measurements")
        return obs, {"step": self.steps, "timestamp": points["time"], "measurements": points}

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Configure the test window in the environment configuration")
        super().reset(seed=seed)
        self.close()
        cfg = self.config
        # 服务端按 JSON 解析请求，案例选择也要发送空对象。
        self.testid = self.request("POST", "/testcases/" + cfg["case"] + "/select", {})["testid"]
        self.request("PUT", f"/step/{self.testid}", {"step": cfg["step_seconds"]})
        points = self.request("PUT", f"/initialize/{self.testid}",
                              {"start_time": cfg["start_time"], "warmup_period": cfg["warmup_seconds"]})
        self.steps = 0
        sample = self._sample(points)
        self.done = False
        return sample

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before step")
        if not self.action_space.contains(action):
            raise ValueError("Action 0 selects embedded control")
        # 失败后需要重新初始化，避免重试推进同一个时间步。
        self.done = True
        points = self.request("POST", f"/advance/{self.testid}", {})
        self.steps += 1
        obs, info = self._sample(points)
        expected = self.config["start_time"] + self.steps * self.config["step_seconds"]
        if info["timestamp"] != expected:
            raise RuntimeError(f"Unexpected BOPTEST time: {info['timestamp']} != {expected}")
        self.done = self.steps >= self.config["episode_steps"]
        return obs, 0.0, False, self.done, info

    def close(self):
        self.done = True
        if self.testid is not None:
            self.request("PUT", f"/stop/{self.testid}", {})
            self.testid = None
