import unittest

import numpy as np
import pandas as pd

from app.indicators.volume_divergence_mm import VolumeDivergenceMM


class VolumeDivergenceIndicatorEMATests(unittest.TestCase):
    def make_data(self, bars=220):
        idx = pd.date_range("2024-01-01", periods=bars, freq="D")
        close = 50 + np.sin(np.arange(bars) / 7) * 3 + np.arange(bars) * 0.03
        return pd.DataFrame({
            "Open": close - np.cos(np.arange(bars)) * 0.4,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 100000 + (np.arange(bars) % 17) * 1500 + np.arange(bars) * 80,
        }, index=idx)

    def test_emas_are_calculated_on_volume_divergence_series(self):
        data = self.make_data()
        result = VolumeDivergenceMM(ema_short=5, ema_medium=20, ema_long=50).calculate(data)
        indicator = result["Volume_Divergence_MM"]
        expected = indicator.ewm(span=5, adjust=False, min_periods=5).mean()
        pd.testing.assert_series_equal(result["Volume_Div_EMA_Short"], expected,
                                       check_names=False)
        self.assertIn("Volume_Div_EMA_Medium", result.columns)
        self.assertIn("Volume_Div_EMA_Long", result.columns)
        self.assertIn("Volume_Div_EMA_Short_Slope", result.columns)

    def test_ema_periods_must_be_in_increasing_order(self):
        with self.assertRaises(ValueError):
            VolumeDivergenceMM(ema_short=20, ema_medium=5, ema_long=50)

    def test_ema_periods_are_configurable(self):
        result = VolumeDivergenceMM(ema_short=3, ema_medium=10, ema_long=30).calculate(
            self.make_data()
        )
        expected = result["Volume_Divergence_MM"].ewm(
            span=3, adjust=False, min_periods=3
        ).mean()
        pd.testing.assert_series_equal(result["Volume_Div_EMA_Short"], expected,
                                       check_names=False)


if __name__ == "__main__":
    unittest.main()
