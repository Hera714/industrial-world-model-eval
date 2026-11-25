"""验证单样本与批量接口、非法输入和三种检查点往返。"""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from models.dynamics_base import PersistenceDynamics
from models.action_mlp import ActionMLP
from models.system_identification import LinearDynamics


class DynamicsAPITest(unittest.TestCase):
    def test_models(self):
        rng = np.random.default_rng(42)
        x, u, d = rng.normal(size=(20,2)), rng.normal(size=(20,1)), rng.normal(size=(20,2))
        linear = LinearDynamics(exogenous_dim=2).fit(x,u,d,x+.1)
        mlp = ActionMLP(hidden=8)
        with self.assertRaises(RuntimeError):
            mlp.predict_next(x,u,d)
        # 仅测试网络序列化和前向契约，训练验收在任务 4 执行。
        mlp.is_fitted = True
        for model, suffix in ((PersistenceDynamics(),'.json'),(linear,'.npz'),(mlp,'.pt')):
            with self.subTest(model=type(model).__name__), tempfile.TemporaryDirectory() as folder:
                prediction = model.predict_next(x,u,d)
                self.assertEqual(prediction.shape, x.shape)
                np.testing.assert_allclose(prediction[0], model.predict_next(x[0],u[0],d[0]), atol=1e-6)
                with self.assertRaises(ValueError):
                    model.predict_next(x, np.full((20,1),np.nan), d)
                with self.assertRaises(ValueError):
                    model.predict_next(x, np.zeros((20,3)), d)
                path = Path(folder)/('model'+suffix)
                model.save(path)
                np.testing.assert_allclose(prediction, type(model).load(path).predict_next(x,u,d), atol=1e-6)
