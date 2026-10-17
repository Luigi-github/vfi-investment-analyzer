"""Regression tests for weekly EMA optimization helpers."""
import unittest

import numpy as np
import pandas as pd

from scripts.optimize_ema_periods import candidate_periods, simulate_strategy, summarize_returns, training_score


class EMAOptimizationTests(unittest.TestCase):
    def test_candidate_grid_contains_5_20_50_and_keeps_ordering(self):
        periods = candidate_periods()
        self.assertIn((5, 20, 50), periods)
        self.assertTrue(periods)
        self.assertTrue(all(0 < short < medium < long for short, medium, long in periods))

    def test_simulated_position_uses_next_bar_after_signal(self):
        index = pd.date_range("2010-01-01", periods=120, freq="W-FRI")
        close = pd.Series(np.linspace(50.0, 100.0, len(index)), index=index)
        vfi = pd.Series(np.linspace(-10.0, 0.0, len(index)), index=index)
        data = pd.DataFrame({"Close": close, "VFI": vfi}, index=index)

        result = simulate_strategy(data, short=3, medium=10, long=20, near_zero=5.0)
        signal_dates = result.index[result["Signal"].eq(1)]
        self.assertGreater(len(signal_dates), 0)
        first_signal = signal_dates[0]
        self.assertEqual(result.loc[first_signal, "Position"], 0.0)
        next_date = result.index[result.index.get_loc(first_signal) + 1]
        self.assertEqual(result.loc[next_date, "Position"], 1.0)


    def test_summary_includes_buy_and_hold_benchmark(self):
        index = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
        close = pd.Series(np.linspace(100.0, 120.0, len(index)), index=index)
        data = pd.DataFrame({
            "Close": close,
            "Position": np.zeros(len(index)),
            "Strategy_Return": np.zeros(len(index)),
            "Signal": np.zeros(len(index), dtype=int),
        }, index=index)
        metrics = summarize_returns(data)
        self.assertAlmostEqual(metrics["BuyHold_Total_Return_Pct"], 20.0)
        self.assertEqual(metrics["Total_Return_Pct"], 0.0)

    def test_training_score_rejects_too_few_entries(self):
        metrics = {"Bars": 200, "Entries": 2, "CAGR_Pct": 15.0, "Max_Drawdown_Pct": -10.0}
        self.assertEqual(training_score(metrics), -np.inf)

    def test_invalid_ema_order_is_rejected(self):
        index = pd.date_range("2010-01-01", periods=80, freq="W-FRI")
        data = pd.DataFrame({
            "Close": np.linspace(10, 20, len(index)),
            "VFI": np.linspace(-2, 2, len(index)),
        }, index=index)
        with self.assertRaisesRegex(ValueError, "short < medium < long"):
            simulate_strategy(data, short=20, medium=5, long=50)


if __name__ == "__main__":
    unittest.main()
