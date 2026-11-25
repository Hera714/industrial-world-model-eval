import unittest
import numpy as np
from metrics.prediction import mae, rmse, error_by_horizon

class MetricsTest(unittest.TestCase):
    def test_known_values(self):
        actual = np.array([[1, 2], [3, 4]])
        predicted = np.array([[2, 4], [2, 4]])
        self.assertAlmostEqual(mae(actual, predicted), 1.0)
        self.assertAlmostEqual(rmse(actual, predicted), np.sqrt(1.5))
        a, r = error_by_horizon(actual, predicted)
        np.testing.assert_allclose(a, [1, 1])
        np.testing.assert_allclose(r, [1, np.sqrt(2)])

    def test_bad_values_rejected(self):
        for a, b in [(np.ones((2, 2)), np.ones((2, 1))),
                     (np.ones((1, 2)), np.array([[1, np.nan]])),
                     (np.array([]), np.array([]))]:
            with self.assertRaises(ValueError): mae(a, b)

if __name__ == "__main__": unittest.main()
