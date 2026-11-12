import unittest

import numpy as np
import pandas as pd

from scripts.build_us_three_rankings import (
    build_combined_ranking,
    require_monthly_ema_history,
)


class TestCombinedRankingMissingMonthlyData(unittest.TestCase):
    def test_missing_monthly_score_is_not_treated_as_zero_percentile(self):
        weekly = pd.DataFrame({
            "Symbol": ["AAA", "BBB", "CCC"],
            "Opportunity_Rank": [1, 2, 3],
            "Weekly_Opportunity_Score": [90.0, 50.0, 10.0],
            "Weekly_VFI": [-1.0, -2.0, 1.0],
            "Weekly_Candle_Date": ["2026-10-23"] * 3,
        })
        monthly = pd.DataFrame({
            "Symbol": ["AAA", "BBB"],
            "Monthly_Rank": [1, 2],
            "Monthly_Score": [10.0, 2.0],
            "Monthly_Divergence": [-100.0, 50.0],
            "Monthly_Candle_Date": ["2026-09-30"] * 2,
        })

        result = build_combined_ranking(weekly, monthly).set_index("Symbol")

        self.assertEqual(result.loc["AAA", "Combined_Status"], "COMPLETE")
        self.assertAlmostEqual(result.loc["AAA", "Monthly_Percentile"], 100.0)
        self.assertAlmostEqual(result.loc["AAA", "Combined_Score"], 100.0)
        self.assertEqual(result.loc["BBB", "Combined_Status"], "COMPLETE")
        self.assertAlmostEqual(result.loc["BBB", "Monthly_Percentile"], 0.0)
        self.assertAlmostEqual(result.loc["BBB", "Combined_Score"], 25.0)

        self.assertEqual(result.loc["CCC", "Combined_Status"], "MISSING_MONTHLY_DATA")
        self.assertTrue(np.isnan(result.loc["CCC", "Monthly_Percentile"]))
        self.assertTrue(np.isnan(result.loc["CCC", "Combined_Score"]))
        self.assertTrue(np.isnan(result.loc["CCC", "Combined_Rank"]))

    def test_monthly_divergence_requires_80_valid_bars_for_ema80(self):
        with self.assertRaisesRegex(ValueError, "need >= 80"):
            require_monthly_ema_history(79)
        require_monthly_ema_history(80)


if __name__ == "__main__":
    unittest.main()
