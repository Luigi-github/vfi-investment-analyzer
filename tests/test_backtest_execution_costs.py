import unittest

import pandas as pd

from app.backtest.execution_costs import LongOnlyExecutionCostModel
from app.backtest.vfi_cycle_backtest import VFICycleBacktest


class TestBacktestExecutionCosts(unittest.TestCase):

    def test_cost_adjusted_forward_return_is_available(self):
        dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
        data = pd.DataFrame(
            {"Close": [100.0, 105.0, 110.0, 120.0]},
            index=dates,
        )

        backtest = VFICycleBacktest(
            horizons=(2,),
            execution_cost_model=LongOnlyExecutionCostModel(
                fee_bps=100,
                slippage_bps=50,
            ),
        )

        result = backtest.calculate_forward_returns(data)

        self.assertAlmostEqual(result.loc[dates[0], "Return_2W"], 10.0)
        self.assertLess(
            result.loc[dates[0], "Net_Return_2W"],
            result.loc[dates[0], "Return_2W"],
        )

    def test_default_backtest_remains_gross_only(self):
        dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
        data = pd.DataFrame(
            {"Close": [100.0, 110.0, 120.0]},
            index=dates,
        )

        result = VFICycleBacktest(horizons=(1,)).calculate_forward_returns(data)

        self.assertIn("Return_1W", result.columns)
        self.assertNotIn("Net_Return_1W", result.columns)


if __name__ == "__main__":
    unittest.main()
