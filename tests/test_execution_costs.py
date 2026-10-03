import unittest

import numpy as np
import pandas as pd

from app.backtest.execution_costs import LongOnlyExecutionCostModel


class TestExecutionCosts(unittest.TestCase):

    def test_zero_costs_preserve_gross_return(self):
        model = LongOnlyExecutionCostModel()
        self.assertAlmostEqual(model.net_return(25.0), 25.0)

    def test_round_trip_costs_reduce_return(self):
        model = LongOnlyExecutionCostModel(
            fee_bps=100,
            slippage_bps=50,
        )

        result = model.net_return(10.0)

        self.assertLess(result, 10.0)
        self.assertGreater(result, 0.0)

    def test_series_preserves_missing_values(self):
        model = LongOnlyExecutionCostModel(fee_bps=10)

        result = model.net_return(
            pd.Series([10.0, np.nan, -5.0])
        )

        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertLess(result.iloc[0], 10.0)
        self.assertLess(result.iloc[2], -5.0)


if __name__ == "__main__":
    unittest.main()
