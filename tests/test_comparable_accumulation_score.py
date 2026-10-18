"""Unit tests for the scale-aware, causal accumulation ranking helpers."""
import unittest

import numpy as np
import pandas as pd

from scripts.study_early_vfi_accumulation import (
    build_asset_summary,
    ema_combinations,
    historical_near_zero_threshold,
    historical_percentile,
    evaluate_combo,
)


class ComparableAccumulationScoreTests(unittest.TestCase):
    def test_ema_grid_has_only_ordered_combinations(self):
        combos = ema_combinations()
        self.assertEqual(len(combos), 64)
        self.assertTrue(all(short < medium < long for short, medium, long in combos))

    def test_near_zero_threshold_uses_prior_history_not_current_value(self):
        prior = pd.Series(np.arange(1.0, 101.0))
        a = historical_near_zero_threshold(pd.concat([prior, pd.Series([0.01])]), 200, 50)
        b = historical_near_zero_threshold(pd.concat([prior, pd.Series([1_000_000.0])]), 200, 50)
        self.assertTrue(np.isclose(a, b))
        self.assertGreater(a, 0)

    def test_historical_percentile_excludes_current_observation(self):
        prior = pd.Series(np.arange(100.0))
        hist_a = pd.concat([prior, pd.Series([50.0])])
        hist_b = pd.concat([prior, pd.Series([10_000.0])])
        p_a = historical_percentile(50.0, hist_a, 200, 50)
        p_b = historical_percentile(50.0, hist_b, 200, 50)
        self.assertTrue(np.isclose(p_a, p_b))
        self.assertGreaterEqual(p_a, 0.0)
        self.assertLessEqual(p_a, 1.0)

    def test_evaluate_combo_returns_bounded_score_on_synthetic_data(self):
        weekly_index = pd.date_range("2018-01-05", periods=400, freq="W-FRI")
        daily_index = pd.date_range("2024-01-01", periods=500, freq="B")
        weekly = pd.DataFrame({"VFI": 2.0 * np.sin(np.arange(400) / 10.0)}, index=weekly_index)
        daily = pd.DataFrame(
            {"Volume_Divergence_MM": 100.0 * np.sin(np.arange(500) / 8.0)},
            index=daily_index,
        )
        result = evaluate_combo("SYNTH", weekly, daily, 3, 15, 40)
        self.assertGreaterEqual(result["Accumulation_Score"], 0.0)
        self.assertLessEqual(result["Accumulation_Score"], 100.0)
        self.assertIn("Weekly_Filter_Pass", result)
        self.assertIn("Daily_Filter_Pass", result)
        component_sum = (
            result["Weekly_State_Score"] + result["Weekly_EMA_Score"]
            + result["Daily_Divergence_Score"] + result["Daily_EMA_Score"]
            + result["Persistence_Score"]
        )
        self.assertAlmostEqual(component_sum, result["Accumulation_Score"], places=3)

    def test_asset_summary_separates_score_from_eligibility(self):
        rows = []
        for symbol, score, eligible in [("A", 80.0, True), ("B", 95.0, False)]:
            rows.append({
                "Symbol": symbol, "Accumulation_Score": score,
                "Weekly_State_Score": 20.0, "Weekly_EMA_Score": 15.0,
                "Daily_Divergence_Score": 20.0, "Daily_EMA_Score": 10.0,
                "Persistence_Score": 10.0, "Weekly_Filter_Pass": eligible,
                "Daily_Filter_Pass": eligible, "Final_Filter_Pass": eligible,
                "EMA_short": 3, "EMA_medium": 15, "EMA_long": 40,
                "VFI_130": -1.0, "VFI_Slope_1W": 0.2,
                "VFI_Slope_Historical_Percentile": 0.7,
                "VFI_Near_Zero_Threshold": 2.0, "Weekly_VFI_Near_Zero": eligible,
                "Weekly_VFI_Positive_Slope": eligible, "Weekly_EMA_Structure_Pass": eligible,
                "Volume_Divergence_MM": -0.2, "Volume_Div_Slope_1D": 0.1,
                "Divergence_Slope_Historical_Percentile": 0.7,
                "Divergence_Near_Zero_Threshold": 1.0, "Daily_Divergence_Near_Zero": eligible,
                "Daily_Divergence_Positive_Slope": eligible, "Daily_EMA_Structure_Pass": eligible,
                "VFI_Negative_Positive_Slope_Streak_Weeks": 2,
                "Divergence_Negative_Positive_Slope_Streak_Days": 3,
                "Weekly_Date": "2026-10-16", "Daily_Date": "2026-10-16",
            })
        summary = build_asset_summary(pd.DataFrame(rows)).set_index("Symbol")
        self.assertTrue(summary.loc["A", "Eligible"])
        self.assertFalse(summary.loc["B", "Eligible"])
        self.assertGreater(summary.loc["B", "Accumulation_Score"], summary.loc["A", "Accumulation_Score"])


if __name__ == "__main__":
    unittest.main()
