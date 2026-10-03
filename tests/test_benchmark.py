import unittest

import pandas as pd

from app.backtest.benchmark import BenchmarkAnalyzer
from app.backtest.execution_costs import LongOnlyExecutionCostModel


class TestBenchmarkAnalyzer(unittest.TestCase):

    def test_event_returns_are_compared_with_all_eligible_dates(self):
        dates = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
        data = pd.DataFrame(
            {"Close": [100.0, 110.0, 90.0, 120.0, 80.0]},
            index=dates,
        )

        result = BenchmarkAnalyzer(horizons=(2,)).compare_event_dates(
            data,
            [dates[0], dates[1]],
        )

        row = result.iloc[0]

        self.assertEqual(row["Event_Count"], 2)
        self.assertEqual(row["Benchmark_Count"], 3)
        self.assertAlmostEqual(row["Event_Average_Return"], -4.0909090909)
        self.assertAlmostEqual(row["Benchmark_Average_Return"], -3.3333333333)

    def test_costs_apply_to_event_and_benchmark_outcomes(self):
        dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
        data = pd.DataFrame(
            {"Close": [100.0, 110.0, 120.0, 130.0]},
            index=dates,
        )

        analyzer = BenchmarkAnalyzer(
            horizons=(1,),
            execution_cost_model=LongOnlyExecutionCostModel(
                fee_bps=100,
                slippage_bps=50,
            ),
        )

        result = analyzer.compare_event_dates(data, [dates[0]])
        row = result.iloc[0]

        self.assertLess(row["Event_Average_Return"], 10.0)
        self.assertLess(row["Benchmark_Average_Return"], 10.0)


if __name__ == "__main__":
    unittest.main()
