import unittest
import numpy as np
import pandas as pd

from scripts.run_colombia_three_rankings import aggregate_bars, ema_opportunity_score, rank_percentile


class ColombiaThreeRankingTests(unittest.TestCase):
    def test_empty_calendar_period_repeats_full_ohlcv(self):
        daily = pd.DataFrame({
            "Open": [10.0, 12.0], "High": [11.0, 13.0],
            "Low": [9.0, 11.0], "Close": [10.5, 12.5],
            "Volume": [100.0, 200.0],
        }, index=pd.to_datetime(["2024-01-05", "2024-01-19"]))
        weekly = aggregate_bars(daily, "W-FRI")
        gap = weekly.loc[pd.Timestamp("2024-01-12")]
        self.assertTrue(bool(gap["Synthetic_Bar"]))
        self.assertEqual(gap["Open"], 10.0)
        self.assertEqual(gap["Close"], 10.5)
        self.assertEqual(gap["Volume"], 100.0)

    def test_ema_is_calculated_on_indicator_series(self):
        idx = pd.date_range("2020-01-03", periods=240, freq="W-FRI")
        indicator = pd.Series(np.linspace(-8, 2, len(idx)), index=idx)
        result, latest = ema_opportunity_score(indicator, {5: .1, 20: .3, 80: .6}, "Monthly")
        self.assertIn("Monthly_EMA_80", result)
        self.assertTrue(np.isfinite(latest))

    def test_rank_percentile_is_scale_independent(self):
        scores = pd.Series([1000.0, 2.0, -100.0], index=["a", "b", "c"])
        pct = rank_percentile(scores)
        self.assertGreater(pct["a"], pct["b"])
        self.assertGreater(pct["b"], pct["c"])


if __name__ == "__main__":
    unittest.main()
