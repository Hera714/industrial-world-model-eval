import unittest
import pandas as pd
from data.split import split_by_time, target_windows, validate_target_windows

class TimeSplitTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100, freq="10min"), "value": range(100)})

    def test_order_counts_and_reproducibility(self):
        first = split_by_time(self.df.sample(frac=1, random_state=2))
        second = split_by_time(self.df)
        self.assertEqual([len(first[k]) for k in ("train", "val", "test")], [70, 15, 15])
        for key in first:
            pd.testing.assert_frame_equal(first[key], second[key])
        windows = target_windows(first, history=5, horizon=3)
        self.assertEqual(windows["val"][0][0], first["val"].date.iloc[0])
        self.assertEqual(len(windows["test"]), 13)

    def test_detects_overlapping_or_out_of_split_targets(self):
        splits = split_by_time(self.df)
        windows = target_windows(splits, history=5, horizon=3)
        windows["val"].append(windows["train"][-1])
        with self.assertRaises(ValueError):
            validate_target_windows(splits, windows)

    def test_duplicate_timestamps_rejected(self):
        duplicate = pd.concat([self.df, self.df.iloc[[0]]])
        with self.assertRaises(ValueError):
            split_by_time(duplicate)

if __name__ == "__main__":
    unittest.main()
