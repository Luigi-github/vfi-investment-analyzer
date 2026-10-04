import unittest

import numpy as np
import pandas as pd

from app.backtest.multi_asset_recovery_analysis import MultiAssetRecoveryAnalyzer


class TestMultiAssetWalkForward(unittest.TestCase):

    def test_condition_discovery_runs_per_asset(self):
        index = pd.date_range("2020-01-03", periods=12, freq="W-FRI")
        events = pd.DataFrame(
            {
                "VFI": np.linspace(-8.0, 2.0, 12),
                "VFI_Slope": [1.0] * 6 + [-1.0] * 6,
                "VFI_Slope_4W": [1.0] * 6 + [-1.0] * 6,
                "VFI_Slope_8W": [1.0] * 6 + [-1.0] * 6,
                "VFI_Distance_Zero": np.linspace(8.0, 2.0, 12),
                "VFI_Range_Position": np.linspace(0.05, 0.95, 12),
                "Return_2W": np.arange(1.0, 13.0),
            },
            index=index,
        )

        analyzer = MultiAssetRecoveryAnalyzer(
            horizons=(2,),
            primary_horizon=2,
        )
        result = analyzer.analyze_condition_discovery_walk_forward(
            {"AAA": events, "BBB": events},
            initial_train_weeks=4,
            test_weeks=2,
            step_weeks=2,
            minimum_events=2,
        )

        self.assertEqual(set(result["summary"]["Symbol"]), {"AAA", "BBB"})
        self.assertTrue((result["summary"]["Window"] >= 1).all())
        self.assertIn("Selected_Condition", result["summary"].columns)

    def test_walk_forward_runs_per_asset_and_preserves_boundaries(self):
        index = pd.date_range("2020-01-03", periods=8, freq="W-FRI")
        events = pd.DataFrame(
            {"Return_2W": np.arange(1.0, 9.0)},
            index=index,
        )

        analyzer = MultiAssetRecoveryAnalyzer(
            horizons=(2,),
            primary_horizon=2,
        )

        result = analyzer.analyze_assets_walk_forward(
            {"AAA": events, "BBB": events},
            initial_train_weeks=2,
            test_weeks=2,
            step_weeks=2,
        )

        self.assertEqual(set(result["Symbol"]), {"AAA", "BBB"})
        self.assertGreaterEqual(result["Window"].max(), 2)
        self.assertTrue(
            (result["Test_Events"] > 0).all()
        )
        self.assertTrue(
            (result["Train_Events"] >= 0).all()
        )

        aaa = result[result["Symbol"] == "AAA"].sort_values("Window")
        self.assertTrue(
            aaa["Cutoff"].is_monotonic_increasing
        )


if __name__ == "__main__":
    unittest.main()
