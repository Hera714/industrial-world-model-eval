import unittest
import numpy as np
import pandas as pd
from data.preprocess import build_dataset, prepare_frame

class PreprocessTest(unittest.TestCase):
    def data(self):
        n = 100
        return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="10min"),
                             "T1": np.arange(n, dtype=float), "T2": np.arange(n, dtype=float) * 2})

    def test_history_targets_and_train_only_scaler(self):
        df = self.data()
        sets, meta = build_dataset(df, history=4, horizon=3, features=["T1", "T2"])
        self.assertAlmostEqual(meta["scaler"]["mean"]["T1"], 34.5)
        x = sets["val"]["X"][0, :, 0] * meta["scaler"]["scale"]["T1"] + 34.5
        y = sets["val"]["y"][0] * meta["scaler"]["scale"]["T1"] + 34.5
        np.testing.assert_allclose(x, [66, 67, 68, 69], atol=1e-5)
        np.testing.assert_allclose(y, [70, 71, 72], atol=1e-5)
        df.loc[70:, "T1"] += 10000
        _, changed = build_dataset(df, history=4, horizon=3, features=["T1", "T2"])
        self.assertEqual(meta["scaler"], changed["scaler"])

    def test_missing_is_marked_and_missing_target_rejected(self):
        df = self.data(); df.loc[75, "T1"] = np.nan
        filled, flags, observed, _ = prepare_frame(df, ["T1", "T2"])
        self.assertEqual(filled.loc[75, "T1"], 74)
        self.assertEqual(flags.loc[75, "T1_missing"], 1)
        self.assertFalse(observed[75])
        sets, meta = build_dataset(df, history=4, horizon=3, features=["T1", "T2"])
        self.assertGreater(meta["splits"]["val"]["discarded_windows"], 0)
        self.assertTrue(sets["val"]["missing_mask"].any())

    def test_off_grid_rejected_and_gap_marked(self):
        df = self.data().drop(index=5)
        _, flags, _, info = prepare_frame(df, ["T1", "T2"])
        self.assertEqual(info["inserted_timestamps"], 1)
        self.assertEqual(flags.loc[5, "T1_missing"], 1)
        df = self.data(); df.loc[5, "date"] += pd.Timedelta(minutes=1)
        with self.assertRaises(ValueError): prepare_frame(df, ["T1", "T2"])

if __name__ == "__main__": unittest.main()
