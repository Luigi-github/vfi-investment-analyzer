import unittest
import numpy as np
import pandas as pd

from scripts.run_colombia_three_rankings import aggregate_bars, ema_opportunity_score, rank_percentile, weekly_gap_metrics


class ColombiaThreeRankingTests(unittest.TestCase):
    def test_empty_calendar_period_carries_prices_but_keeps_zero_volume(self):
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
        self.assertEqual(gap["Volume"], 0.0)

    def test_short_history_is_not_eligible(self):
        daily = pd.DataFrame({
            "Open": [10.0, 12.0], "High": [11.0, 13.0],
            "Low": [9.0, 11.0], "Close": [10.5, 12.5],
            "Volume": [100.0, 200.0],
        }, index=pd.to_datetime(["2024-01-05", "2024-01-19"]))
        weekly = aggregate_bars(daily, "W-FRI")
        self.assertFalse(weekly_gap_metrics(weekly)["Gap_Pass"])

    def test_zero_volume_calendar_week_is_treated_as_gap(self):
        daily = pd.DataFrame({
            "Open": [10.0, 10.0, 12.0], "High": [11.0, 11.0, 13.0],
            "Low": [9.0, 9.0, 11.0], "Close": [10.5, 10.5, 12.5],
            "Volume": [100.0, 0.0, 200.0],
        }, index=pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"]))
        weekly = aggregate_bars(daily, "W-FRI")
        self.assertEqual(weekly_gap_metrics(weekly)["Total_Missing_Weeks"], 1)

    def test_sparse_old_gap_passes_but_latest_four_weeks_are_protected(self):
        idx = pd.date_range("2024-01-05", periods=60, freq="W-FRI")
        weekly = pd.DataFrame({
            "Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.5,
            "Volume": 100.0, "Synthetic_Bar": False,
        }, index=idx)
        weekly.loc[idx[-20], "Synthetic_Bar"] = True
        weekly.loc[idx[-20], "Volume"] = 0.0
        metrics = weekly_gap_metrics(weekly)
        self.assertTrue(metrics["Gap_Pass"])
        self.assertEqual(metrics["Recent_Missing_Weeks"], 1)
        weekly.loc[idx[-2], "Volume"] = 0.0
        self.assertFalse(weekly_gap_metrics(weekly)["Gap_Pass"])

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
