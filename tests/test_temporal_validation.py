import unittest

import numpy as np
import pandas as pd

from app.backtest.temporal_validation import TemporalHoldoutAnalyzer


class TestTemporalHoldoutAnalyzer(unittest.TestCase):

    def setUp(self):
        index = pd.date_range("2020-01-03", periods=5, freq="W-FRI")
        self.events = pd.DataFrame(
            {
                "Return_2W": [10.0, -5.0, 20.0, np.nan, 5.0],
            },
            index=index,
        )

    def test_split_requires_train_labels_to_end_before_cutoff(self):
        analyzer = TemporalHoldoutAnalyzer(primary_horizon=2)

        train, test = analyzer.split(
            self.events,
            "2020-01-31",
        )

        label_end = train.index + pd.to_timedelta(2, unit="W")

        self.assertTrue((label_end < pd.Timestamp("2020-01-31")).all())
        self.assertTrue((test.index >= pd.Timestamp("2020-01-31")).all())
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 1)

    def test_summaries_use_only_available_forward_returns(self):
        analyzer = TemporalHoldoutAnalyzer(primary_horizon=2)

        result = analyzer.analyze(
            self.events,
            "2020-01-31",
        )

        self.assertEqual(result["train_summary"]["Events"], 2)
        self.assertAlmostEqual(
            result["train_summary"]["Average_Return"],
            2.5,
        )
        self.assertEqual(result["test_summary"]["Events"], 1)
        self.assertAlmostEqual(
            result["test_summary"]["Average_Return"],
            5.0,
        )
        self.assertAlmostEqual(
            result["test_summary"]["Positive_Rate"],
            100.0,
        )

    def test_multi_asset_summary(self):
        analyzer = TemporalHoldoutAnalyzer(primary_horizon=2)

        result = analyzer.analyze_assets(
            {"AAA": self.events, "BBB": self.events.iloc[:3]},
            "2020-01-31",
        )

        self.assertEqual(set(result["Symbol"]), {"AAA", "BBB"})
        self.assertEqual(
            result.loc[result["Symbol"] == "AAA", "Train_Events"].iloc[0],
            2,
        )
        self.assertEqual(
            result.loc[result["Symbol"] == "AAA", "Test_Events"].iloc[0],
            1,
        )

    def test_multiple_walk_forward_windows_have_strict_boundaries(self):
        analyzer = TemporalHoldoutAnalyzer(primary_horizon=2)

        windows = analyzer.walk_forward(
            self.events,
            initial_train_weeks=2,
            test_weeks=2,
            step_weeks=2,
        )

        self.assertGreaterEqual(len(windows), 1)
        self.assertTrue(
            (windows[0]["train"].index
             + pd.to_timedelta(2, unit="W")
             < windows[0]["cutoff"]).all()
        )

        for previous, current in zip(windows, windows[1:]):
            self.assertTrue(
                previous["test"].index.max()
                < current["test"].index.min()
            )

    def test_invalid_index_is_rejected(self):
        analyzer = TemporalHoldoutAnalyzer()

        events = self.events.reset_index(drop=True)

        with self.assertRaises(ValueError):
            analyzer.split(events, "2020-01-31")


if __name__ == "__main__":
    unittest.main()
