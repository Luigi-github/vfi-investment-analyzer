import unittest

import pandas as pd

from app.backtest.multi_asset_cycle_analysis import MultiAssetCycleAnalyzer


def make_asset(offset=0):
    index = pd.date_range("2024-01-05", periods=14, freq="W-FRI")
    return pd.DataFrame(
        {
            "Close": [100 + offset, 102 + offset, 105 + offset, 110 + offset,
                      115 + offset, 120 + offset, 118 + offset, 112 + offset,
                      105 + offset, 103 + offset, 106 + offset, 109 + offset,
                      114 + offset, 118 + offset],
            "VFI": [-3, -2, -1, 0.5, 2, 3, 2, 1, 0, -1, -2, -1, 0.5, 2],
            "VFI_Slope": [0, 1, 1, 1.5, 1.5, 1, -1, -1, -1, 1, 1, 1, 1.5, 1],
            "VFI_Slope_4W": [0, 0.2, 0.5, 0.8, 1, 1, 0.5, -0.5, -1, -0.5, 0.2, 0.5, 0.8, 1],
            "VFI_Slope_8W": [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.4, -0.2, -0.6, -0.3, 0.1, 0.3, 0.5, 0.6],
            "VFI_Cycle_Stage": [
                "EARLY_RECOVERY", "SUSTAINED_RECOVERY", "SUSTAINED_RECOVERY",
                "ZERO_CROSS_UP", "ACCUMULATION", "BULLISH_FLOW", "WEAKENING",
                "ZERO_CROSS_DOWN", "BEARISH_FLOW", "EARLY_RECOVERY",
                "SUSTAINED_RECOVERY", "APPROACHING_ZERO", "ZERO_CROSS_UP",
                "ACCUMULATION",
            ],
        },
        index=index,
    )


class TestMultiAssetCycleAnalyzer(unittest.TestCase):

    def test_analyzes_multiple_assets(self):
        result = MultiAssetCycleAnalyzer().analyze(
            {"DDD": make_asset(), "INTC": make_asset(10)}
        )

        self.assertFalse(result["cycles"].empty)
        self.assertFalse(result["events"].empty)
        self.assertEqual(set(result["cycles"]["Symbol"]), {"DDD", "INTC"})
        self.assertEqual(set(result["events"]["Symbol"]), {"DDD", "INTC"})
        self.assertFalse(result["stage_statistical_summary"].empty)
        self.assertIn("Bootstrap_CI_Lower", result["stage_statistical_summary"].columns)

    def test_primary_horizon_is_configurable(self):
        analyzer = MultiAssetCycleAnalyzer(horizons=(4, 8), primary_horizon=4)
        result = analyzer.analyze({"DDD": make_asset()})

        self.assertIn("Average_Return_4W", result["stage_summary"].columns)


if __name__ == "__main__":
    unittest.main()
