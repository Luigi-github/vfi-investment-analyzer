import unittest

import numpy as np
import pandas as pd

from app.indicators.volume_divergence_mm import VolumeDivergenceMM


class VolumeDivergenceMMTests(unittest.TestCase):
    def test_pine_wma_applies_candle_direction_and_weights_latest_candle_more(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="D")
        data = pd.DataFrame(
            {
                "Open": [10.0, 12.0, 11.0],
                "High": [11.0, 13.0, 12.0],
                "Low": [9.0, 11.0, 10.0],
                "Close": [11.0, 11.0, 12.0],
                "Volume": [10.0, 20.0, 30.0],
            },
            index=idx,
        )
        indicator = VolumeDivergenceMM()
        result = indicator._pine_wma(data["Volume"], data["Open"], data["Close"], 3)

        # Oldest -> newest observations receive weights 1, 2, 3.
        expected = (10 * 1 * 1 + 20 * 2 * -1 + 30 * 3 * 1) / 6
        self.assertTrue(np.isclose(result.iloc[-1], expected))

    def test_pivot_confirmation_is_not_backdated(self):
        indicator = VolumeDivergenceMM(lbL=2, lbR=2)
        values = np.array([5.0, 4.0, 1.0, 3.0, 5.0, 6.0])
        pivots = indicator._pivot(values, 2, 2, high=False)

        self.assertFalse(pivots[2])
        self.assertFalse(pivots[3])
        self.assertTrue(pivots[4])

    def test_indicator_outputs_causal_divergence_columns(self):
        idx = pd.date_range("2025-01-01", periods=120, freq="D")
        close = 100 + np.sin(np.arange(120) / 5.0) * 5 + np.arange(120) * 0.02
        open_ = close - 0.5
        high = np.maximum(open_, close) + 1
        low = np.minimum(open_, close) - 1
        volume = 1000 + (np.arange(120) % 10) * 25

        data = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=idx,
        )
        result = VolumeDivergenceMM().calculate(data)

        required = {
            "Volume_Divergence_MM",
            "Volume_Div_Regular_Bull",
            "Volume_Div_Hidden_Bull",
            "Volume_Div_Regular_Bear",
            "Volume_Div_Hidden_Bear",
            "Volume_Div_Bullish",
            "Volume_Div_Bearish",
            "Volume_Div_Bullish_Visible",
            "Volume_Div_Bearish_Visible",
            "Volume_Div_Visible_Signal",
            "Volume_Div_Bullish_Visible_Age_Bars",
            "Volume_Div_Bearish_Visible_Age_Bars",
            "Volume_Div_Signal",
            "Volume_Div_Bullish_Age_Bars",
            "Volume_Div_Bearish_Age_Bars",
        }
        self.assertTrue(required.issubset(result.columns))


    def test_visible_signals_match_tradingview_hidden_divergence_defaults(self):
        idx = pd.date_range("2025-01-01", periods=120, freq="D")
        close = 100 + np.sin(np.arange(120) / 5.0) * 5 + np.arange(120) * 0.02
        open_ = close - 0.5
        data = pd.DataFrame({
            "Open": open_,
            "High": np.maximum(open_, close) + 1,
            "Low": np.minimum(open_, close) - 1,
            "Close": close,
            "Volume": 1000 + (np.arange(120) % 10) * 25,
        }, index=idx)
        default = VolumeDivergenceMM().calculate(data)
        self.assertTrue((default["Volume_Div_Bullish_Visible"] == default["Volume_Div_Regular_Bull"]).all())
        self.assertTrue((default["Volume_Div_Bearish_Visible"] == default["Volume_Div_Regular_Bear"]).all())

        enabled = VolumeDivergenceMM(plot_hidden_bull=True, plot_hidden_bear=True).calculate(data)
        self.assertTrue((enabled["Volume_Div_Bullish_Visible"] == enabled["Volume_Div_Bullish"]).all())
        self.assertTrue((enabled["Volume_Div_Bearish_Visible"] == enabled["Volume_Div_Bearish"]).all())


if __name__ == "__main__":
    unittest.main()
