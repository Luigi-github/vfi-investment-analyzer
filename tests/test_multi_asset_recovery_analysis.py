import unittest

import numpy as np
import pandas as pd

from app.backtest.multi_asset_recovery_analysis import MultiAssetRecoveryAnalyzer


class TestMultiAssetRecoveryAnalyzer(unittest.TestCase):
    def make_data(self):
        dates = pd.date_range("2020-01-03", periods=70, freq="W-FRI")
        stages = ["ACCUMULATION"] * len(dates)
        for index in (5, 6, 20, 21):
            stages[index] = "EARLY_RECOVERY"

        return pd.DataFrame(
            {
                "Close": np.linspace(100.0, 140.0, len(dates)),
                "VFI": np.linspace(-10.0, 10.0, len(dates)),
                "VFI_Slope": np.ones(len(dates)),
                "VFI_Slope_4W": np.ones(len(dates)),
                "VFI_Slope_8W": np.ones(len(dates)),
                "VFI_Distance_Zero": np.abs(np.linspace(-10.0, 10.0, len(dates))),
                "VFI_Range_Position": np.full(len(dates), 0.5),
                "VFI_Cycle_Stage": stages,
            },
            index=dates,
        )

    def test_causal_analysis_handles_assets_without_recovery_events(self):
        analyzer = MultiAssetRecoveryAnalyzer(primary_horizon=26)
        data = self.make_data()
        data["VFI_Cycle_Stage"] = "ACCUMULATION"

        result = analyzer.analyze_asset_causal("TEST", data)

        self.assertTrue(result.empty)
        self.assertIn("Recovery_Cluster_ID", analyzer._ensure_causal_recovery_clusters(data).columns)

    def test_causal_analysis_builds_clusters_when_missing(self):
        analyzer = MultiAssetRecoveryAnalyzer(primary_horizon=26)
        data = self.make_data()

        clustered = analyzer._ensure_causal_recovery_clusters(data)

        self.assertIn("Recovery_Cluster_ID", clustered.columns)
        self.assertEqual(
            clustered.loc[clustered["VFI_Cycle_Stage"].eq("EARLY_RECOVERY"),
                           "Recovery_Cluster_ID"].tolist(),
            [1, 1, 2, 2],
        )

        result = analyzer.analyze_asset_causal("TEST", data)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["Representative_Method"], "CAUSAL_FIRST_RECOVERY")
        self.assertTrue(bool(result.iloc[0]["Representative_Causal"]))


if __name__ == "__main__":
    unittest.main()
