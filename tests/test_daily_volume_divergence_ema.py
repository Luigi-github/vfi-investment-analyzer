import unittest

import numpy as np
import pandas as pd

from scripts.optimize_daily_volume_divergence_ema import simulate_strategy, summarize_returns


class DailyVolumeDivergenceEMATests(unittest.TestCase):
    def make_data(self):
        idx = pd.date_range("2024-01-01", periods=150, freq="D")
        close = np.linspace(50.0, 100.0, len(idx))
        frame = pd.DataFrame({
            "Open": close - 0.2, "High": close + 1.0, "Low": close - 1.0,
            "Close": close, "Volume_Div_Regular_Bull": False,
            "Volume_Div_Regular_Bear": False,
        }, index=idx)
        return frame

    def test_entry_requires_recent_bullish_divergence_and_ema_alignment(self):
        frame = self.make_data()
        idx = frame.index
        frame.loc[idx[100], "Volume_Div_Regular_Bull"] = True
        result = simulate_strategy(frame, 3, 10, 20, divergence_fresh_bars=5)
        self.assertEqual(result.loc[idx[100], "Position"], 0.0)
        self.assertEqual(result.loc[idx[101], "Position"], 1.0)
        self.assertEqual(result.loc[idx[105], "Position"], 1.0)

    def test_bearish_divergence_exits_on_next_bar(self):
        frame = self.make_data()
        idx = frame.index
        frame.loc[idx[100], "Volume_Div_Regular_Bull"] = True
        frame.loc[idx[110], "Volume_Div_Regular_Bear"] = True
        result = simulate_strategy(frame, 3, 10, 20, divergence_fresh_bars=5)
        self.assertEqual(result.loc[idx[101], "Position"], 1.0)
        self.assertEqual(result.loc[idx[110], "Position"], 1.0)
        self.assertEqual(result.loc[idx[111], "Position"], 0.0)

    def test_summary_includes_buy_and_hold_benchmark(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        close = np.linspace(100.0, 120.0, len(idx))
        frame = pd.DataFrame({
            "Close": close, "Position": np.zeros(len(idx)),
            "Strategy_Return": np.zeros(len(idx)), "Signal": np.zeros(len(idx), dtype=int),
        }, index=idx)
        metrics = summarize_returns(frame)
        self.assertAlmostEqual(metrics["BuyHold_Total_Return_Pct"], 20.0)
        self.assertEqual(metrics["Total_Return_Pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
