"""Unit tests for the scale-aware, causal accumulation ranking helpers."""
import unittest

import numpy as np
import pandas as pd

from scripts.study_early_vfi_accumulation import (
    build_asset_summary,
    _indicator_state,
    ema_combinations,
    historical_near_zero_threshold,
    historical_percentile,
    positive_long_ema_penalty,
    descending_slope_penalty,
    descending_ema_penalty,
    vfi_smoothness_reward,
    positive_negative_streak,
    evaluate_combo,
    weekly_vfi_trend_penalty,
    vfi_positive_extension_penalty,
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

    def test_positive_indicator_cannot_earn_early_accumulation_slope_points(self):
        history = pd.Series(np.arange(1.0, 201.0))
        slopes = pd.Series(np.arange(1.0, 201.0))
        state = _indicator_state(
            value=100.0, slope=200.0, value_history=history, slope_history=slopes,
            lookback=252, min_periods=100,
        )
        self.assertFalse(state["negative"])
        self.assertFalse(state["near_zero"])
        self.assertEqual(state["score"], 0.0)

    def test_positive_vfi_penalty_increases_with_historical_extension(self):
        history = pd.Series(np.linspace(-10.0, 10.0, 200))
        low = vfi_positive_extension_penalty(2.0, pd.concat([history, pd.Series([2.0])]), 252, 100)
        high = vfi_positive_extension_penalty(9.0, pd.concat([history, pd.Series([9.0])]), 252, 100)
        negative = vfi_positive_extension_penalty(-9.0, pd.concat([history, pd.Series([-9.0])]), 252, 100)
        self.assertGreaterEqual(low["VFI_Positive_Extension_Penalty"], 0.0)
        self.assertGreater(high["VFI_Positive_Extension_Penalty"], low["VFI_Positive_Extension_Penalty"])
        self.assertLessEqual(high["VFI_Positive_Extension_Penalty"], 15.0)
        self.assertEqual(negative["VFI_Positive_Extension_Penalty"], 0.0)
        self.assertIn("VFI_Historical_Percentile", high)

    def test_vfi_extension_penalty_uses_raw_historical_percentile(self):
        history_a = pd.Series(np.linspace(-10.0, 10.0, 200))
        history_b = history_a * 1000.0
        penalty_a = vfi_positive_extension_penalty(9.0, pd.concat([history_a, pd.Series([9.0])]), 252, 100)
        penalty_b = vfi_positive_extension_penalty(9000.0, pd.concat([history_b, pd.Series([9000.0])]), 252, 100)
        self.assertAlmostEqual(penalty_a["VFI_Positive_Extension_Penalty"], penalty_b["VFI_Positive_Extension_Penalty"])

    def test_positive_long_ema_gets_very_high_penalty(self):
        positive = positive_long_ema_penalty(0.01)
        zero = positive_long_ema_penalty(0.0)
        negative = positive_long_ema_penalty(-0.01)
        self.assertTrue(positive["VFI_Long_EMA_Positive"])
        self.assertEqual(positive["VFI_Long_EMA_Positive_Penalty"], 25.0)
        self.assertEqual(zero["VFI_Long_EMA_Positive_Penalty"], 0.0)
        self.assertEqual(negative["VFI_Long_EMA_Positive_Penalty"], 0.0)

    def test_vfi_extension_threshold_uses_prior_bars_only(self):
        prior = pd.Series(np.linspace(-20.0, 20.0, 200))
        low_current = vfi_positive_extension_penalty(10.0, pd.concat([prior, pd.Series([10.0])]), 252, 100)
        huge_current = vfi_positive_extension_penalty(10.0, pd.concat([prior, pd.Series([1_000_000.0])]), 252, 100)
        self.assertEqual(
            low_current["VFI_Positive_Extension_Penalty"],
            huge_current["VFI_Positive_Extension_Penalty"],
        )

    def test_historical_percentile_excludes_current_observation(self):
        prior = pd.Series(np.arange(100.0))
        hist_a = pd.concat([prior, pd.Series([50.0])])
        hist_b = pd.concat([prior, pd.Series([10_000.0])])
        p_a = historical_percentile(50.0, hist_a, 200, 50)
        p_b = historical_percentile(50.0, hist_b, 200, 50)
        self.assertTrue(np.isclose(p_a, p_b))
        self.assertGreaterEqual(p_a, 0.0)
        self.assertLessEqual(p_a, 1.0)

    def test_evaluate_combo_score_includes_uncapped_streak_adjustment(self):
        weekly_index = pd.date_range("2018-01-05", periods=400, freq="W-FRI")
        daily_index = pd.date_range("2024-01-01", periods=500, freq="B")
        weekly = pd.DataFrame({"VFI": 2.0 * np.sin(np.arange(400) / 10.0)}, index=weekly_index)
        daily = pd.DataFrame(
            {"Volume_Divergence_MM": 100.0 * np.sin(np.arange(500) / 8.0)},
            index=daily_index,
        )
        result = evaluate_combo("SYNTH", weekly, daily, 3, 15, 40)
        # Final score is intentionally unbounded and includes the raw weekly streak adjustment.
        self.assertTrue(np.isfinite(result["Accumulation_Score"]))
        self.assertIn("VFI_Weekly_Streak_Adjustment", result)
        self.assertIn("Weekly_Filter_Pass", result)
        self.assertIn("Daily_Filter_Pass", result)
        component_sum = (
            result["Weekly_State_Score"] + result["Weekly_EMA_Score"]
            + result["Daily_Divergence_Score"] + result["Daily_EMA_Score"]
            + result["Persistence_Score"] - result["Weekly_Trend_Penalty"]
            - result["VFI_Positive_Extension_Penalty"]
            - result["VFI_Long_EMA_Positive_Penalty"]
            - result["VFI_Descending_Slope_Penalty"]
            - result["VFI_Total_Descending_EMA_Penalty"]
            - result["Divergence_Total_Descending_EMA_Penalty"]
            - result["VFI_EMA_200_Descending_Penalty"]
            + result["VFI_Smoothness_Reward"]
            + result["VFI_Weekly_Streak_Adjustment"]
            + result["VFI_Relative_Position_Adjustment"]
        )
        self.assertAlmostEqual(component_sum, result["Accumulation_Score"], places=3)

    def test_penalizes_bearish_ema_recross_and_vfi_uptrend_loss(self):
        frame = pd.DataFrame({
            "VFI": [-3.0, -2.5, -2.0, -1.5, -1.6],
            "EMA_short": [-2.8, -2.3, -1.8, -1.4, -1.7],
            "EMA_medium": [-3.0, -2.4, -1.9, -1.5, -1.6],
            "EMA_long": [-3.5, -3.0, -2.8, -2.0, -1.65],
            "Indicator_Slope_1Bar": [0.2, 0.5, 0.5, 0.5, -0.1],
        })
        penalty = weekly_vfi_trend_penalty(frame)
        self.assertTrue(penalty["Weekly_EMA_Bearish_Recross"])
        self.assertTrue(penalty["VFI_Uptrend_Lost"])
        self.assertEqual(penalty["Weekly_EMA_Short_Medium_Cross_Penalty"], 20.0)
        self.assertEqual(penalty["Weekly_EMA_Short_Long_Cross_Penalty"], 20.0)
        self.assertEqual(penalty["Weekly_EMA_Medium_Long_Cross_Penalty"], 0.0)
        self.assertEqual(penalty["Weekly_EMA_Bearish_Recross_Penalty"], 40.0)
        self.assertEqual(penalty["VFI_Uptrend_Loss_Penalty"], 7.0)
        self.assertEqual(penalty["Weekly_Trend_Penalty"], 47.0)

    def test_no_trend_penalty_when_no_bearish_recross_or_uptrend_loss(self):
        frame = pd.DataFrame({
            "VFI": [-3.0, -2.8, -2.6, -2.4, -2.2],
            "EMA_short": [-2.9, -2.7, -2.5, -2.3, -2.1],
            "EMA_medium": [-3.1, -2.9, -2.7, -2.5, -2.3],
            "EMA_long": [-3.5, -3.2, -3.0, -2.8, -2.6],
            "Indicator_Slope_1Bar": [0.2, 0.2, 0.2, 0.2, 0.2],
        })
        penalty = weekly_vfi_trend_penalty(frame)
        self.assertFalse(penalty["Weekly_EMA_Bearish_Recross"])
        self.assertFalse(penalty["Weekly_EMA_Short_Below_Long_Recross"])
        self.assertFalse(penalty["Weekly_EMA_Medium_Below_Long_Recross"])
        self.assertFalse(penalty["VFI_Uptrend_Lost"])
        self.assertEqual(penalty["Weekly_Trend_Penalty"], 0.0)


    def test_descending_slope_is_penalized_more_than_flat_or_rising(self):
        scale = 10.0
        falling = descending_slope_penalty(pd.Series([0, -1, -2, -3, -4], dtype=float), scale, 4, 5.0)
        flat = descending_slope_penalty(pd.Series([-2, -2, -2, -2, -2], dtype=float), scale, 4, 5.0)
        rising = descending_slope_penalty(pd.Series([-4, -3, -2, -1, 0], dtype=float), scale, 4, 5.0)
        self.assertGreater(falling, 0.0)
        self.assertEqual(flat, 0.0)
        self.assertEqual(rising, 0.0)

    def test_descending_penalties_cover_each_ema_curve(self):
        frame = pd.DataFrame({
            "EMA_short": [-1, -2, -3, -4],
            "EMA_medium": [-1, -1.5, -2, -2.5],
            "EMA_long": [-1, -1.2, -1.4, -1.6],
        }, dtype=float)
        result = descending_ema_penalty(frame, scale=10.0)
        self.assertGreater(result["EMA_short_Descending_Penalty"], 0.0)
        self.assertGreater(result["EMA_medium_Descending_Penalty"], 0.0)
        self.assertGreater(result["EMA_long_Descending_Penalty"], 0.0)
        self.assertAlmostEqual(
            result["Total_Descending_EMA_Penalty"],
            result["EMA_short_Descending_Penalty"]
            + result["EMA_medium_Descending_Penalty"]
            + result["EMA_long_Descending_Penalty"],
        )

    def test_longer_negative_rising_vfi_streak_gets_more_persistence(self):
        short = pd.Series([-4.0, -3.0, -2.0])
        long = pd.Series(np.linspace(-8.0, -1.0, 27))
        self.assertEqual(positive_negative_streak(short), 2)
        self.assertEqual(positive_negative_streak(long), 26)
        self.assertGreater(
            8.0 * min(positive_negative_streak(long) / 26.0, 1.0),
            8.0 * min(positive_negative_streak(short) / 26.0, 1.0),
        )

    def test_each_vfi_ema_streak_requires_negative_and_rising(self):
        rising_negative = pd.Series(np.linspace(-10.0, -1.0, 27))
        falling_negative = pd.Series(np.linspace(-1.0, -10.0, 27))
        crosses_zero = pd.Series(np.linspace(-10.0, 1.0, 27))
        self.assertEqual(positive_negative_streak(rising_negative), 26)
        self.assertEqual(positive_negative_streak(falling_negative), 0)
        self.assertEqual(positive_negative_streak(crosses_zero), 0)

    def test_vfi_smoothness_reward_requires_rising_negative_vfi(self):
        smooth_rising_negative = pd.Series(np.linspace(-2.0, -0.2, 12))
        choppy_rising_negative = pd.Series([-2.0, -1.0, -1.8, -0.8, -1.4, -0.6, -1.0, -0.5, -0.9, -0.4, -0.7, -0.2])
        falling_negative = pd.Series(np.linspace(-0.2, -2.0, 12))
        already_positive = pd.Series(np.linspace(0.2, 2.0, 12))
        smooth = vfi_smoothness_reward(smooth_rising_negative, scale=10.0)
        choppy = vfi_smoothness_reward(choppy_rising_negative, scale=10.0)
        falling = vfi_smoothness_reward(falling_negative, scale=10.0)
        positive = vfi_smoothness_reward(already_positive, scale=10.0)
        self.assertGreater(smooth["VFI_Smoothness_Reward"], 0.0)
        deep_negative = vfi_smoothness_reward(pd.Series(np.linspace(-8.0, -6.2, 12)), scale=10.0)
        self.assertGreater(smooth["VFI_Smoothness_Reward"], choppy["VFI_Smoothness_Reward"])
        self.assertGreater(deep_negative["VFI_Smoothness_Reward"], 0.0)
        self.assertGreater(smooth["VFI_Smoothness_Reward"], deep_negative["VFI_Smoothness_Reward"])
        self.assertEqual(falling["VFI_Smoothness_Reward"], 0.0)
        self.assertEqual(positive["VFI_Smoothness_Reward"], 0.0)
        self.assertGreaterEqual(smooth["VFI_Trend_R2"], 0.99)

    def test_asset_summary_separates_score_from_eligibility(self):
        rows = []
        for symbol, score, eligible in [("A", 80.0, True), ("B", 95.0, False)]:
            rows.append({
                "Symbol": symbol, "Accumulation_Score": score,
                "Weekly_State_Score": 20.0, "Weekly_EMA_Score": 15.0,
                "Daily_Divergence_Score": 20.0, "Daily_EMA_Score": 10.0,
                "Persistence_Score": 10.0, "Gross_Accumulation_Score": score,
                "Weekly_EMA_Bearish_Recross": False,
                "Weekly_EMA_Short_Below_Medium_Recross": False,
                "Weekly_EMA_Short_Below_Long_Recross": False,
                "Weekly_EMA_Medium_Below_Long_Recross": False,
                "VFI_Uptrend_Lost": False,
                "Weekly_EMA_Bearish_Recross_Penalty": 0.0,
                "Weekly_EMA_Short_Medium_Cross_Penalty": 0.0,
                "Weekly_EMA_Short_Long_Cross_Penalty": 0.0,
                "Weekly_EMA_Medium_Long_Cross_Penalty": 0.0,
                "VFI_Uptrend_Loss_Penalty": 0.0, "Weekly_Trend_Penalty": 0.0,
                "VFI_Positive_Extended": False, "VFI_Historical_Percentile": 0.5,
                "VFI_Extension_Threshold": 1.0, "VFI_Extreme_Threshold": 2.0,
                "VFI_Positive_Extension_Penalty": 0.0, "VFI_Long_EMA_Positive": False,
                "VFI_Long_EMA_Positive_Penalty": 0.0,
                "VFI_Slope_1W_Penalty": 0.0, "VFI_Slope_4W_Penalty": 0.0,
                "VFI_Slope_8W_Penalty": 0.0, "VFI_Descending_Slope_Penalty": 0.0,
                "VFI_EMA_short_Descending_Penalty": 0.0,
                "VFI_EMA_medium_Descending_Penalty": 0.0,
                "VFI_EMA_long_Descending_Penalty": 0.0,
                "VFI_Total_Descending_EMA_Penalty": 0.0,
                "Divergence_EMA_short_Descending_Penalty": 0.0,
                "Divergence_EMA_medium_Descending_Penalty": 0.0,
                "Divergence_EMA_long_Descending_Penalty": 0.0,
                "Divergence_Total_Descending_EMA_Penalty": 0.0,
                "VFI_Trend_R2": 0.8, "VFI_Trend_Regression_Slope": 0.2,
                "VFI_Smoothness_Reward": 0.0, "Total_Score_Penalty": 0.0,
                "Weekly_Filter_Pass": eligible,
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
