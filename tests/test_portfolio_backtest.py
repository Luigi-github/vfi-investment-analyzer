import unittest

import numpy as np
import pandas as pd

from app.backtest.portfolio_backtest import PortfolioBacktester


class TestPortfolioBacktester(unittest.TestCase):
    def make_prices(self):
        dates = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
        a = np.linspace(100.0, 160.0, len(dates))
        b = np.linspace(100.0, 130.0, len(dates))
        return {
            "A": pd.DataFrame({"Close": a}, index=dates),
            "B": pd.DataFrame({"Close": b}, index=dates),
        }

    def make_signals(self):
        dates = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
        return pd.DataFrame(
            {"Symbol": ["A", "B"], "Signal_Score": [2.0, 1.0]},
            index=[dates[5], dates[6]],
        )

    def test_equal_weight_portfolio_generates_equity_curve(self):
        result = PortfolioBacktester(
            horizon_weeks=8,
            max_positions=2,
            weighting="equal",
        ).run(self.make_prices(), self.make_signals())

        self.assertFalse(result["equity_curve"].empty)
        self.assertGreater(result["summary"]["Total_Return"], 0)
        self.assertLessEqual(
            result["equity_curve"]["Exposure"].max(), 1.0 + 1e-9
        )

    def test_max_positions_is_respected(self):
        prices = self.make_prices()
        dates = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
        signals = pd.DataFrame(
            {
                "Symbol": ["A", "B", "A", "B"],
                "Signal_Score": [1.0, 2.0, 3.0, 4.0],
            },
            index=[dates[5], dates[5], dates[6], dates[6]],
        )

        result = PortfolioBacktester(
            horizon_weeks=8,
            max_positions=1,
            weighting="equal",
        ).run(prices, signals)

        self.assertTrue((result["equity_curve"]["Active_Positions"] <= 1).all())
        self.assertTrue(
            (result["trades"]["Weight"] <= 1.0 + 1e-9).all()
        )

    def test_inverse_vol_and_risk_parity_are_supported(self):
        for weighting in ("inverse_vol", "risk_parity"):
            result = PortfolioBacktester(
                horizon_weeks=8,
                max_positions=2,
                weighting=weighting,
                volatility_lookback=12,
                min_history_weeks=4,
            ).run(self.make_prices(), self.make_signals())

            self.assertFalse(result["equity_curve"].empty)
            self.assertTrue(np.isfinite(result["summary"]["Total_Return"]))

    def test_costs_reduce_or_equal_net_performance(self):
        gross = PortfolioBacktester(
            horizon_weeks=8,
            max_positions=2,
            weighting="equal",
        ).run(self.make_prices(), self.make_signals())

        net = PortfolioBacktester(
            horizon_weeks=8,
            max_positions=2,
            weighting="equal",
            fee_bps=25,
            slippage_bps=25,
        ).run(self.make_prices(), self.make_signals())

        self.assertLessEqual(
            net["summary"]["Total_Return"],
            gross["summary"]["Total_Return"] + 1e-9,
        )
        self.assertGreater(
            net["equity_curve"]["Transaction_Cost"].sum(),
            0,
        )


if __name__ == "__main__":
    unittest.main()
