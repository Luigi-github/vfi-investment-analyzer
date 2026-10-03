import unittest

import numpy as np
import pandas as pd

from app.backtest.multi_asset_recovery_analysis import MultiAssetRecoveryAnalyzer


class TestMultiAssetWalkForward(unittest.TestCase):

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
