import unittest

import numpy as np
import pandas as pd

from scripts.vfi_accumulation_divergence_study import (
    add_divergence_cycle_features,
    classify_two_stage_opportunities,
    discover_crypto_universe,
)


class TwoStageOpportunityFilterTests(unittest.TestCase):
    @staticmethod
    def candidate_frame():
        return pd.DataFrame(
            {
                "VFI": [-0.4, 1.8, -0.2, 0.1, -0.5],
                "VFI_Early_Setup": [True, False, True, False, True],
                "VFI_Zero_Cross_Up_Recent_Near_Zero": [False, False, False, True, False],
                "Weekly_Div_Osc_Early_Setup": [True, True, False, False, True],
                "Weekly_Div_Osc_Zero_Cross_Up_Recent": [False, False, False, True, False],
                "Weekly_Div_Value": [-0.3, -0.2, 0.6, 0.1, -0.2],
                "Weekly_Divergence_History_Sufficient": [True, True, True, True, False],
                "VFI_Slope_10D_Normalized": [0.5, 0.8, 0.4, 0.5, 0.4],
                "Weekly_Div_Osc_Slope_2W_Normalized": [0.6, 0.4, 0.3, 0.5, 0.4],
                "Weekly_Bullish_Div_Recent": [False, True, False, False, False],
            },
            index=["both_pre_zero", "vfi_late", "vfi_only", "both_crossed", "no_weekly_history"],
        )

    def test_only_vfi_first_then_divergence_is_an_actionable_candidate(self):
        result = classify_two_stage_opportunities(self.candidate_frame())
        self.assertTrue(bool(result.loc["both_pre_zero", "Two_Stage_Opportunity"]))
        self.assertEqual(result.loc["both_pre_zero", "Opportunity_Tier"], 1)
        self.assertEqual(result.loc["both_pre_zero", "Candidate_Status"], "TWO_STAGE_CANDIDATE")

        self.assertFalse(bool(result.loc["vfi_late", "Two_Stage_Opportunity"]))
        self.assertEqual(result.loc["vfi_late", "VFI_Stage"], "POSITIVE_AFTER_CROSS_OR_LATE")
        self.assertEqual(result.loc["vfi_late", "Candidate_Status"], "DIVERGENCE_READY_WAITING_FOR_VFI")

        self.assertFalse(bool(result.loc["vfi_only", "Two_Stage_Opportunity"]))
        self.assertEqual(result.loc["vfi_only", "Candidate_Status"], "VFI_READY_WAITING_FOR_DIVERGENCE")

        self.assertTrue(bool(result.loc["both_crossed", "Two_Stage_Opportunity"]))
        self.assertEqual(result.loc["both_crossed", "Opportunity_Tier"], 4)

        self.assertEqual(result.loc["no_weekly_history", "Divergence_Cycle_Stage"], "NO_HISTORY")
        self.assertFalse(bool(result.loc["no_weekly_history", "Two_Stage_Opportunity"]))

    def test_weekly_oscillator_setup_requires_negative_near_zero_strong_recovery(self):
        idx = pd.date_range("2022-01-02", periods=130, freq="W-SUN")
        osc = np.concatenate([np.linspace(-15.0, -10.0, 100), np.linspace(-9.5, -0.3, 30)])
        frame = pd.DataFrame({"Volume_Divergence_MM": osc}, index=idx)
        result = add_divergence_cycle_features(
            frame, periods=(1, 2, 4), suffix="W", scale_window=52, min_periods=20
        )
        self.assertTrue(bool(result["Div_Osc_Negative_Near_Zero_W"].iloc[-1]))
        self.assertTrue(bool(result["Div_Osc_Strong_Rising_Slope_W"].iloc[-1]))
        self.assertTrue(bool(result["Div_Osc_Early_Setup_W"].iloc[-1]))
        self.assertGreater(result["Div_Osc_Estimated_Weeks_To_Zero_W"].iloc[-1], 0)
        self.assertLessEqual(result["Div_Osc_Estimated_Weeks_To_Zero_W"].iloc[-1], 8)

    def test_late_positive_divergence_oscillator_is_not_pre_zero_setup(self):
        idx = pd.date_range("2022-01-02", periods=130, freq="W-SUN")
        osc = np.linspace(1.0, 10.0, len(idx))
        frame = pd.DataFrame({"Volume_Divergence_MM": osc}, index=idx)
        result = add_divergence_cycle_features(frame)
        self.assertFalse(bool(result["Div_Osc_Early_Setup_W"].iloc[-1]))
        self.assertEqual(result["Div_Osc_Estimated_Weeks_To_Zero_W"].iloc[-1], np.nan)

    def test_dynamic_universe_ranks_intersection_by_combined_liquidity_rank(self):
        class FakeService:
            SPOT_BASE = "spot"
            FUTURES_BASE = "perp"

            def discover_spot_usdt(self, minimum):
                return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

            def discover_usdt_perpetuals(self, minimum):
                return ["SOLUSDT", "BTCUSDT", "ETHUSDT"]

            def _get(self, base, path, params):
                symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
                if base == "spot":
                    return [{"symbol": s, "quoteVolume": "100", "priceChangePercent": "1"} for s in symbols]
                return [{"symbol": s, "quoteVolume": "200", "priceChangePercent": "2"} for s in symbols]

        result = discover_crypto_universe(FakeService(), limit=2)
        self.assertEqual([row["Symbol"] for row in result], ["BTCUSDT", "SOLUSDT"])
        self.assertEqual([row["Universe_Rank"] for row in result], [1, 2])
        self.assertEqual(result[0]["Combined_Volume_Rank_Sum"], 3)


if __name__ == "__main__":
    unittest.main()
