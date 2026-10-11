import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_volume_accumulation_indicators import calculate_indicators


def make_ohlcv(n=120, trend=1.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.arange(n) * trend * 0.1 + np.sin(np.arange(n) / 4)
    volume = 1000 + (np.arange(n) % 7) * 100
    return pd.DataFrame({
        "Open": close - 0.2,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": volume,
    }, index=idx)


class VolumeAccumulationIndicatorTests(unittest.TestCase):
    def test_indicator_scores_are_bounded_and_composite_exists(self):
        result = calculate_indicators(make_ohlcv())
        for key in ("OBV_Trend_Score", "AD_Trend_Score", "CMF_Score",
                    "MFI_Score", "Accumulation_Distribution_Score"):
            self.assertTrue(0 <= result[key] <= 100, key)
        self.assertEqual(result["Indicator_Count"], 4)
        self.assertEqual(result["Bars_Used"], 120)

    def test_zero_range_does_not_break_accumulation_distribution(self):
        data = make_ohlcv()
        data["High"] = data["Close"]
        data["Low"] = data["Close"]
        result = calculate_indicators(data)
        self.assertTrue(np.isfinite(result["Accumulation_Distribution_Score"]))

    def test_insufficient_history_is_reported(self):
        with self.assertRaisesRegex(ValueError, "Insufficient valid bars"):
            calculate_indicators(make_ohlcv(n=20))


if __name__ == "__main__":
    unittest.main()
