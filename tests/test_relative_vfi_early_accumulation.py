import unittest

import numpy as np
import pandas as pd

from scripts.run_us_relative_vfi_early_accumulation_ranking import (
    historical_percentile,
    score_weekly_vfi,
    trailing_true_streak,
)


class RelativeVfiRankingTests(unittest.TestCase):
    def test_streak_counts_only_current_uninterrupted_run(self):
        self.assertEqual(trailing_true_streak([True, True, False, True, True, True]), 3)
        self.assertEqual(trailing_true_streak([True, False, True, False]), 0)

    def test_percentile_is_asset_relative_and_scale_invariant(self):
        history = pd.Series(np.arange(-50.0, 50.0))
        pct = historical_percentile(history, 10.0)
        scaled_pct = historical_percentile(history * 100.0, 1000.0)
        self.assertAlmostEqual(pct, scaled_pct, places=10)

    def test_score_is_scale_invariant_for_positive_scale(self):
        idx = pd.date_range("2010-01-01", periods=420, freq="W-FRI")
        t = np.arange(len(idx), dtype=float)
        # Negative VFI recovering towards zero, with a small cyclical component.
        values = -18.0 + 0.035 * t + 1.5 * np.sin(t / 8.0)
        original = score_weekly_vfi(pd.Series(values, index=idx))
        scaled = score_weekly_vfi(pd.Series(values * 25.0, index=idx))
        self.assertAlmostEqual(original["Final_Score"], scaled["Final_Score"], places=7)
        self.assertAlmostEqual(
            original["VFI_Absolute_Magnitude_Percentile"],
            scaled["VFI_Absolute_Magnitude_Percentile"],
            places=7,
        )

    def test_score_contains_transparent_components_and_penalties(self):
        idx = pd.date_range("2010-01-01", periods=420, freq="W-FRI")
        t = np.arange(len(idx), dtype=float)
        values = -10.0 + 0.02 * t + np.sin(t / 6.0)
        result = score_weekly_vfi(pd.Series(values, index=idx))
        for key in (
            "Final_Score", "EMA_Persistence_Score", "Relative_Position_Score",
            "VFI_Persistence_Score", "Acceleration_Score", "Total_Penalty",
            "EMA200_Rising_Streak_Weeks", "VFI_Acceleration_Percentile",
        ):
            self.assertIn(key, result)
        self.assertGreaterEqual(result["Final_Score"], 0.0)
        self.assertLessEqual(result["Final_Score"], 100.0)


    def test_near_zero_is_diagnostic_not_a_scoring_component(self):
        idx = pd.date_range("2010-01-01", periods=420, freq="W-FRI")
        t = np.arange(len(idx), dtype=float)
        values = -25.0 + 0.04 * t + 0.8 * np.sin(t / 9.0)
        result = score_weekly_vfi(pd.Series(values, index=idx))
        expected = (
            0.30 * result["EMA_Persistence_Score"]
            + 0.25 * result["EMA_Level_Signed_Score"]
            + 0.25 * result["VFI_Persistence_Score"]
            + 0.20 * result["Acceleration_Score"]
        )
        self.assertAlmostEqual(result["Gross_Score"], expected, places=7)
        self.assertIn("VFI_Near_Zero_Score", result)

    def test_negative_ema_adds_and_positive_ema_subtracts_by_percentile(self):
        idx = pd.date_range("2010-01-01", periods=420, freq="W-FRI")
        t = np.arange(len(idx), dtype=float)
        values = -35.0 + 0.055 * t + 0.7 * np.sin(t / 8.0)
        result = score_weekly_vfi(pd.Series(values, index=idx))
        for key in ("EMA5", "EMA20", "EMA80", "EMA200"):
            value = result[f"{key}_Value"]
            percentile = result[f"{key}_Percentile"]
            component = result[f"{key}_Signed_Level_Component"]
            if value < 0:
                self.assertAlmostEqual(component, 100.0 - percentile, places=7)
            elif value > 0:
                self.assertAlmostEqual(component, -percentile, places=7)
            else:
                self.assertEqual(component, 0.0)
        self.assertGreaterEqual(result["EMA_Level_Signed_Score"], 0.0)
        self.assertLessEqual(result["EMA_Level_Signed_Score"], 100.0)

    def test_sustained_pullback_is_penalized_more_than_rising_trend(self):
        idx = pd.date_range("2010-01-01", periods=420, freq="W-FRI")
        t = np.arange(len(idx), dtype=float)
        rising = -30.0 + 0.05 * t + 0.4 * np.sin(t / 8.0)
        falling = rising.copy()
        falling[-26:] -= np.linspace(0.0, 18.0, 26)
        rising_result = score_weekly_vfi(pd.Series(rising, index=idx))
        falling_result = score_weekly_vfi(pd.Series(falling, index=idx))
        self.assertGreater(falling_result["Pullback_Penalty"], rising_result["Pullback_Penalty"])
        self.assertGreater(falling_result["Total_Penalty"], rising_result["Total_Penalty"])

    def test_score_exposes_multi_horizon_deterioration(self):
        idx = pd.date_range("2010-01-01", periods=420, freq="W-FRI")
        t = np.arange(len(idx), dtype=float)
        values = -15.0 + 0.03 * t + np.sin(t / 7.0)
        result = score_weekly_vfi(pd.Series(values, index=idx))
        self.assertIn("VFI_Slope_8W", result)
        self.assertIn("VFI_Slope_26W", result)
        self.assertIn("EMA_Pullback_Penalty", result)


if __name__ == "__main__":
    unittest.main()
