import unittest

import numpy as np
import pandas as pd

from app.research.accumulation_breakout_screen import (
    rank_accumulation_candidates,
    score_zero_approach,
    unavailable_asset_row,
)


def rising_negative_series(end=-0.5, length=60):
    values = -7.0 + np.sin(np.arange(length) / 3.0) * 2.0
    values[-9:] = np.linspace(-5.0, end, 9)
    return pd.Series(values)


class TestAccumulationBreakoutScreen(unittest.TestCase):
    def test_pre_zero_break_requires_negative_near_zero_and_strong_positive_slopes(self):
        result = score_zero_approach(
            rising_negative_series(),
            "VFI_1D",
            lookback=52,
            min_history=20,
            near_zero_fraction=0.15,
            max_abs_near_zero=2.0,
        )
        self.assertEqual(result["VFI_1D_Setup_Type"], "PRE_ZERO_BREAK")
        self.assertLess(result["VFI_1D_Value"], 0)
        self.assertTrue(result["VFI_1D_Near_Zero"])
        self.assertTrue(result["VFI_1D_Strong_Positive_Slope"])
        self.assertGreater(result["VFI_1D_Slope_1B"], 0)
        self.assertGreater(result["VFI_1D_Slope_4B"], 0)
        self.assertGreater(result["VFI_1D_Slope_8B"], 0)

    def test_high_positive_oscillator_is_not_an_early_breakout_candidate(self):
        values = pd.Series(np.linspace(-8.0, 6.0, 60))
        result = score_zero_approach(values, "Divergence_1W", lookback=52, min_history=20)
        self.assertEqual(result["Divergence_1W_Setup_Type"], "NONE")

    def test_recent_zero_cross_is_labelled_separately(self):
        values = rising_negative_series(end=0.1)
        result = score_zero_approach(values, "VFI_1D", lookback=52, min_history=20)
        self.assertEqual(result["VFI_1D_Setup_Type"], "ZERO_CROSS_RECENT")

    def test_zero_cross_two_bars_ago_is_accepted_for_daily_window(self):
        values = -7.0 + np.sin(np.arange(60) / 3.0) * 2.0
        values[-9:] = np.linspace(-5.0, -0.5, 9)
        values[-4:] = [-0.1, 0.05, 0.08, 0.1]
        result = score_zero_approach(
            pd.Series(values), "VFI_1D", recent_cross_bars=3
        )
        self.assertEqual(result["VFI_1D_Setup_Type"], "ZERO_CROSS_RECENT")
        self.assertEqual(result["VFI_1D_Recent_Zero_Cross_Bars_Ago"], 2)

    def test_zero_cross_older_than_allowed_window_is_rejected(self):
        values = -7.0 + np.sin(np.arange(60) / 3.0) * 2.0
        values[-9:] = np.linspace(-5.0, -0.5, 9)
        values[-5:] = [-0.1, 0.05, 0.1, 0.15, 0.2]
        result = score_zero_approach(
            pd.Series(values), "Divergence_1W", recent_cross_bars=2
        )
        self.assertEqual(result["Divergence_1W_Setup_Type"], "NONE")

    def test_recent_zero_cross_with_weak_slope_is_rejected(self):
        values = -7.0 + np.sin(np.arange(60) / 3.0) * 2.0
        values[-9:] = np.linspace(-0.15, 0.05, 9)
        result = score_zero_approach(pd.Series(values), "VFI_1D")
        self.assertEqual(result["VFI_1D_Setup_Type"], "NONE")

    def test_short_cycle_history_keeps_latest_value_and_slopes_for_audit(self):
        values = pd.Series(np.linspace(-4.0, -0.5, 40))
        result = score_zero_approach(
            values, "VFI_1D", lookback=365, min_history=60
        )
        self.assertEqual(result["VFI_1D_Setup_Type"], "INSUFFICIENT_HISTORY")
        self.assertAlmostEqual(result["VFI_1D_Value"], -0.5)
        self.assertGreater(result["VFI_1D_Slope_1B"], 0)
        self.assertGreater(result["VFI_1D_Slope_4B"], 0)
        self.assertGreater(result["VFI_1D_Slope_8B"], 0)
        self.assertTrue(pd.isna(result["VFI_1D_Range_365B"]))
        self.assertIn("need 60", result["VFI_1D_Setup_Reason"])

    def test_insufficient_history_is_not_ranked_as_setup(self):
        result = score_zero_approach(pd.Series([-3.0, -2.0, -1.0]), "VFI_1D")
        self.assertEqual(result["VFI_1D_Setup_Type"], "INSUFFICIENT_HISTORY")
        self.assertEqual(result["VFI_1D_Setup_Score"], 0.0)

    def test_unavailable_history_is_preserved_but_never_a_candidate(self):
        placeholder = unavailable_asset_row(
            "NEWCOINUSDT", 200, "INSUFFICIENT_HISTORY", "not enough daily candles"
        )
        frame = pd.DataFrame(
            [
                {
                    "Symbol": "VALID",
                    "VFI_Setup_Type": "PRE_ZERO_BREAK",
                    "Divergence_Setup_Type": "PRE_ZERO_BREAK",
                    "VFI_Setup_Score": 80.0,
                    "Divergence_Setup_Score": 70.0,
                    "Bullish_Divergence_Recent": False,
                    "Analysis_Status": "ANALYZED",
                },
                placeholder,
            ]
        )
        result = rank_accumulation_candidates(frame)
        self.assertEqual(len(result), 2)
        missing = result.set_index("Symbol").loc["NEWCOINUSDT"]
        self.assertEqual(missing["Analysis_Status"], "INSUFFICIENT_HISTORY")
        self.assertFalse(bool(missing["VFI_Setup"]))
        self.assertFalse(bool(missing["Combined_Setup"]))
        self.assertEqual(int(missing["Universe_Liquidity_Rank"]), 200)

    def test_combined_daily_vfi_weekly_divergence_setup_ranks_first(self):
        frame = pd.DataFrame(
            [
                {
                    "Symbol": "VFI_ONLY",
                    "VFI_Setup_Type": "PRE_ZERO_BREAK",
                    "Divergence_Setup_Type": "NONE",
                    "VFI_Setup_Score": 95.0,
                    "Divergence_Setup_Score": 30.0,
                    "Bullish_Divergence_Recent": False,
                },
                {
                    "Symbol": "BOTH",
                    "VFI_Setup_Type": "PRE_ZERO_BREAK",
                    "Divergence_Setup_Type": "PRE_ZERO_BREAK",
                    "VFI_Setup_Score": 75.0,
                    "Divergence_Setup_Score": 70.0,
                    "Bullish_Divergence_Recent": True,
                },
                {
                    "Symbol": "NONE",
                    "VFI_Setup_Type": "NONE",
                    "Divergence_Setup_Type": "NONE",
                    "VFI_Setup_Score": 50.0,
                    "Divergence_Setup_Score": 50.0,
                    "Bullish_Divergence_Recent": False,
                },
            ]
        )
        result = rank_accumulation_candidates(frame)
        self.assertEqual(result.iloc[0]["Symbol"], "BOTH")
        self.assertTrue(bool(result.iloc[0]["Combined_Setup"]))
        self.assertEqual(result.iloc[1]["Symbol"], "VFI_ONLY")
        self.assertEqual(result.iloc[2]["Symbol"], "NONE")


if __name__ == "__main__":
    unittest.main()
