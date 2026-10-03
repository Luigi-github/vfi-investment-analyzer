import unittest

import numpy as np
import pandas as pd

from app.backtest.temporal_validation import TemporalHoldoutAnalyzer


class TestTemporalHoldoutAnalyzer(unittest.TestCase):

    def setUp(self):
        index = pd.date_range("2020-01-03", periods=5, freq="W-FRI")
        self.events = pd.DataFrame(
            {
                "Return_26W": [10.0, -5.0, 20.0, np.nan, 5.0],
            },
            index=index,
        )

    def test_split_is_strictly_chronological(self):
        analyzer = TemporalHoldoutAnalyzer(primary_horizon=26)

        train, test = analyzer.split(
            self.events,
            "2020-01-17",
        )

        self.assertTrue((train.index < pd.Timestamp("2020-01-17")).all())
        self.assertTrue((test.index >= pd.Timestamp("2020-01-17")).all())
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 3)

    def test_summaries_use_only_available_forward_returns(self):
        analyzer = TemporalHoldoutAnalyzer(primary_horizon=26)

        result = analyzer.analyze(
            self.events,
            "2020-01-17",
        )

        self.assertEqual(result["train_summary"]["Events"], 2)
        self.assertAlmostEqual(
            result["train_summary"]["Average_Return"],
            2.5,
        )
        self.assertEqual(result["test_summary"]["Events"], 2)
        self.assertAlmostEqual(
            result["test_summary"]["Average_Return"],
            12.5,
        )
        self.assertAlmostEqual(
            result["test_summary"]["Positive_Rate"],
            100.0,
        )

    def test_invalid_index_is_rejected(self):
        analyzer = TemporalHoldoutAnalyzer()

        events = self.events.reset_index(drop=True)

        with self.assertRaises(ValueError):
            analyzer.split(events, "2020-01-17")


if __name__ == "__main__":
    unittest.main()
