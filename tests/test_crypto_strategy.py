import unittest
import pandas as pd
from app.backtest.crypto_strategy import backtest_signals

class TestCryptoStrategy(unittest.TestCase):
    def test_entry_is_next_bar(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
        h = pd.DataFrame({
            "Open_Spot": range(100,110), "High_Spot": range(101,111),
            "Low_Spot": range(99,109), "Close_Spot": range(100,110),
            "Score": [80]+[0]*9
        }, index=idx)
        trades, _ = backtest_signals(h, "Score", threshold=70, horizon_hours=3, fee=0, slippage=0)
        self.assertEqual(trades.iloc[0]["Entry_Date"], idx[1])

    def test_stop_loss(self):
        idx = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
        h = pd.DataFrame({
            "Open_Spot": [100,100,100,100,100,100],
            "High_Spot": [101,101,101,101,101,101],
            "Low_Spot": [99,99,94,99,99,99],
            "Close_Spot": [100]*6, "Score": [80,0,0,0,0,0]
        }, index=idx)
        trades, _ = backtest_signals(h, "Score", threshold=70, horizon_hours=3, fee=0, slippage=0, stop_loss=.05)
        self.assertEqual(trades.iloc[0]["Exit_Reason"], "STOP")
        self.assertAlmostEqual(trades.iloc[0]["Net_Return"], -.05)

if __name__ == "__main__":
    unittest.main()
