import unittest

import numpy as np
import pandas as pd

from app.analysis.vfi_analyzer import VFIAnalyzer
from app.indicators.vfi import VFIIndicator


class TestVFISlopeUnits(unittest.TestCase):
    def sample_ohlcv(self, bars=220):
        index = pd.date_range("2025-01-01", periods=bars, freq="D", tz="UTC")
        close = 100 + np.arange(bars) * 0.15 + np.sin(np.arange(bars) / 4.0)
        open_ = close - 0.1
        return pd.DataFrame(
            {
                "Open": open_,
                "High": close + 0.8,
                "Low": open_ - 0.8,
                "Close": close,
                "Volume": 1000 + (np.arange(bars) % 13) * 25,
            },
            index=index,
        )

    def test_indicator_exposes_bar_based_slopes_and_legacy_aliases(self):
        result = VFIIndicator(context_window=52).calculate(self.sample_ohlcv())
        valid = result["VFI"].dropna().index[-1]
        position = result.index.get_loc(valid)
        expected_4 = (result["VFI"].iloc[position] - result["VFI"].iloc[position - 4]) / 4.0
        expected_8 = (result["VFI"].iloc[position] - result["VFI"].iloc[position - 8]) / 8.0

        self.assertIn("VFI_Slope_4B", result.columns)
        self.assertIn("VFI_Slope_8B", result.columns)
        self.assertAlmostEqual(result.loc[valid, "VFI_Slope_4B"], expected_4)
        self.assertAlmostEqual(result.loc[valid, "VFI_Slope_8B"], expected_8)
        self.assertAlmostEqual(result.loc[valid, "VFI_Slope_4W"], expected_4)
        self.assertAlmostEqual(result.loc[valid, "VFI_Slope_8W"], expected_8)

    def test_analyzer_prefers_bar_based_slopes(self):
        frame = self.sample_ohlcv()
        result = VFIIndicator(context_window=52).calculate(frame)
        analyzed = VFIAnalyzer().analyze(result)
        self.assertIn("VFI_State", analyzed.columns)
        self.assertEqual(len(analyzed), len(frame))

    def test_analyzer_keeps_compatibility_with_legacy_slope_columns(self):
        frame = pd.DataFrame(
            {
                "VFI": [-2.0, -1.5, -1.0],
                "VFI_Slope": [0.5, 0.5, 0.5],
                "VFI_Slope_4W": [0.2, 0.3, 0.4],
                "VFI_Slope_8W": [0.1, 0.2, 0.3],
                "VFI_Distance_Zero": [2.0, 1.5, 1.0],
                "VFI_Rolling_Min": [-3.0, -3.0, -3.0],
                "VFI_Rolling_Max": [1.0, 1.0, 1.0],
            }
        )
        result = VFIAnalyzer().analyze(frame)
        self.assertIn("VFI_State", result.columns)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
