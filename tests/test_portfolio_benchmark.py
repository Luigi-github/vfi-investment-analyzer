import unittest

import numpy as np
import pandas as pd

from app.backtest.portfolio_benchmark import PortfolioBenchmark


class TestPortfolioBenchmark(unittest.TestCase):
    def test_equal_weight_benchmark(self):
        dates = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
        prices = {
            "A": pd.DataFrame({"Close": np.arange(100.0, 110.0)}, index=dates),
            "B": pd.DataFrame({"Close": np.arange(100.0, 120.0, 2.0)}, index=dates),
        }

        curve = PortfolioBenchmark().equal_weight_rebalanced(prices)

        self.assertFalse(curve.empty)
        self.assertEqual(len(curve), 9)
        self.assertGreater(curve.iloc[-1]["Equity"], 1.0)

    def test_single_asset_buy_hold(self):
        dates = pd.date_range("2020-01-03", periods=5, freq="W-FRI")
        frame = pd.DataFrame({"Close": [100, 110, 90, 120, 130]}, index=dates)

        curve = PortfolioBenchmark().single_asset_buy_hold(frame)

        self.assertAlmostEqual(curve.iloc[-1]["Equity"], 1.3)
        self.assertAlmostEqual(curve.iloc[-1]["Drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()
