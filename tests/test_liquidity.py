import unittest

import numpy as np
import pandas as pd

from app.backtest.liquidity import LiquidityAnalyzer


class TestLiquidityAnalyzer(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-03", periods=20, freq="W-FRI")
        self.data = {
            "A": pd.DataFrame(
                {
                    "Close": np.full(len(dates), 100.0),
                    "Volume": np.full(len(dates), 100000.0),
                },
                index=dates,
            )
        }
        self.signals = pd.DataFrame(
            {"Symbol": ["A"]},
            index=[dates[-1]],
        )

    def test_point_in_time_capacity_uses_only_past_data(self):
        analyzer = LiquidityAnalyzer(lookback_weeks=13)
        result = analyzer.annotate_signals(
            self.signals,
            self.data,
            capital=100000,
            max_participation_pct=1.0,
        )

        self.assertAlmostEqual(result.iloc[0]["Avg_Dollar_Volume"], 10000000.0)
        self.assertAlmostEqual(result.iloc[0]["Max_Tradable_Weight"], 1.0)

    def test_capacity_filter_can_reject_signal(self):
        analyzer = LiquidityAnalyzer(lookback_weeks=13)
        result = analyzer.filter_signals(
            self.signals,
            self.data,
            capital=1000000,
            max_participation_pct=0.1,
            target_position_weight=0.10,
        )

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
