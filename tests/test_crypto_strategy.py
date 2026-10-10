import unittest
import pandas as pd
from app.backtest.crypto_strategy import backtest_signals, _max_drawdown


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

    def test_missing_signal_values_do_not_open_trades(self):
        idx = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
        h = pd.DataFrame({
            "Open_Spot": [100.0] * 12,
            "High_Spot": [101.0] * 12,
            "Low_Spot": [99.0] * 12,
            "Close_Spot": [100.0] * 12,
            "Score": [float("nan"), float("nan"), 80.0] + [0.0] * 9,
        }, index=idx)
        trades, _ = backtest_signals(
            h, "Score", threshold=70, horizon_hours=3, fee=0, slippage=0
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["Signal_Date"], idx[2])
        self.assertEqual(trades.iloc[0]["Entry_Date"], idx[3])

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

    def test_funding_is_cost_for_long_and_credit_for_short(self):
        idx = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
        h = pd.DataFrame({
            "Open_Spot": [100]*6,
            "High_Spot": [101]*6,
            "Low_Spot": [99]*6,
            "Close_Spot": [100]*6,
            "Score": [80,0,0,0,0,0],
        }, index=idx)
        funding = pd.Series([0.01], index=[idx[3]])
        long_trades, _ = backtest_signals(
            h, "Score", threshold=70, horizon_hours=3, fee=0, slippage=0,
            direction="long", funding=funding,
        )
        short_trades, _ = backtest_signals(
            h, "Score", threshold=70, horizon_hours=3, fee=0, slippage=0,
            direction="short", funding=funding,
        )
        self.assertAlmostEqual(long_trades.iloc[0]["Funding_PnL"], -0.01)
        self.assertAlmostEqual(short_trades.iloc[0]["Funding_PnL"], 0.01)


    def test_funding_window_excludes_entry_and_includes_exit(self):
        idx = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
        h = pd.DataFrame({
            "Open_Perp": [100.0] * 6,
            "High_Perp": [101.0] * 6,
            "Low_Perp": [99.0] * 6,
            "Close_Perp": [100.0] * 6,
            "Score": [80.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }, index=idx)
        funding = pd.Series([0.10, 0.02], index=[idx[1], idx[4]])
        trades, _ = backtest_signals(
            h, "Score", threshold=70, horizon_hours=3, fee=0, slippage=0,
            direction="long", funding=funding, price_prefix="Perp",
        )
        # Entry is idx[1], exit is idx[4]: funding at entry is excluded.
        self.assertEqual(trades.iloc[0]["Entry_Date"], idx[1])
        self.assertEqual(trades.iloc[0]["Exit_Date"], idx[4])
        self.assertAlmostEqual(trades.iloc[0]["Funding_PnL"], -0.02)

    def test_short_stop_and_target_returns_match_configured_percentages(self):
        idx = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
        base = {
            "Open_Spot": [100.0] * 6,
            "High_Spot": [101.0] * 6,
            "Low_Spot": [99.0] * 6,
            "Close_Spot": [100.0] * 6,
            "Score": [80.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }

        stop_frame = pd.DataFrame(base, index=idx)
        stop_frame.loc[idx[1], "High_Spot"] = 106.0
        stop_trades, _ = backtest_signals(
            stop_frame, "Score", threshold=70, horizon_hours=3,
            fee=0, slippage=0, stop_loss=0.05, take_profit=0.10,
            direction="short",
        )
        self.assertEqual(stop_trades.iloc[0]["Exit_Reason"], "STOP")
        self.assertAlmostEqual(stop_trades.iloc[0]["Gross_Return"], -0.05)
        self.assertAlmostEqual(stop_trades.iloc[0]["Net_Return"], -0.05)

        target_frame = pd.DataFrame(base, index=idx)
        target_frame.loc[idx[1], "Low_Spot"] = 89.0
        target_trades, _ = backtest_signals(
            target_frame, "Score", threshold=70, horizon_hours=3,
            fee=0, slippage=0, stop_loss=0.05, take_profit=0.10,
            direction="short",
        )
        self.assertEqual(target_trades.iloc[0]["Exit_Reason"], "TARGET")
        self.assertAlmostEqual(target_trades.iloc[0]["Gross_Return"], 0.10)
        self.assertAlmostEqual(target_trades.iloc[0]["Net_Return"], 0.10)


    def test_max_drawdown_includes_initial_capital(self):
        equity_after_trades = pd.Series([0.90, 0.95])
        self.assertAlmostEqual(_max_drawdown(equity_after_trades), -0.10)


if __name__ == "__main__":
    unittest.main()
