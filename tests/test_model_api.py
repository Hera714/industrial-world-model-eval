import unittest
import numpy as np
from models.persistence import Persistence
from models.arx import LinearAutoregression
from models.learned_dynamics import MLPForecast

class ModelAPITest(unittest.TestCase):
    def test_all_models_share_shape_and_finite_output(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(40, 5, 2)).astype("float32")
        y = np.repeat(X[:, -1, :1], 3, axis=1)
        models = [Persistence(3), LinearAutoregression((0.1, 1.0)), MLPForecast(horizon=3, epochs=2, patience=2, device="cpu")]
        for model in models:
            model.fit(X[:24], y[:24], X[24:32], y[24:32])
            p = model.predict(X[32:])
            self.assertEqual(p.shape, (8, 3))
            self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(models[0].predict(X[32:]), y[32:])
    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError): Persistence().predict(np.array([[1.0]]))

if __name__ == "__main__": unittest.main()
