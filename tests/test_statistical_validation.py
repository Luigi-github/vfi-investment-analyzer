import unittest

import numpy as np
import pandas as pd

from app.backtest.statistical_validation import StatisticalValidationAnalyzer


class TestStatisticalValidationAnalyzer(unittest.TestCase):
    def test_summary_reports_core_distribution_metrics(self):
        result = StatisticalValidationAnalyzer().summarize([10, -5, 15, 0])
        self.assertEqual(result["Count"], 4)
        self.assertAlmostEqual(result["Average_Return"], 5.0)
        self.assertAlmostEqual(result["Median_Return"], 5.0)
        self.assertAlmostEqual(result["Positive_Rate"], 50.0)

    def test_bootstrap_is_reproducible(self):
        analyzer = StatisticalValidationAnalyzer(bootstrap_samples=200, random_seed=7)
        first = analyzer.bootstrap_mean([1, 2, 3, 4, 5])
        second = analyzer.bootstrap_mean([1, 2, 3, 4, 5])
        self.assertEqual(first, second)
        self.assertLessEqual(first["CI_Lower"], first["Mean"])
        self.assertGreaterEqual(first["CI_Upper"], first["Mean"])

    def test_benchmark_comparison_keeps_event_and_benchmark_separate(self):
        result = StatisticalValidationAnalyzer().compare_to_benchmark(
            [10, 20, -10],
            [5, 5, 5, 5],
        )
        self.assertAlmostEqual(result["Event_Average_Return"], 20 / 3)
        self.assertAlmostEqual(result["Benchmark_Average_Return"], 5.0)
        self.assertAlmostEqual(result["Excess_Average_Return"], 5 / 3)

    def test_nan_returns_are_dropped(self):
        result = StatisticalValidationAnalyzer().summarize(
            pd.Series([1.0, np.nan, -1.0])
        )
        self.assertEqual(result["Count"], 2)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            StatisticalValidationAnalyzer(bootstrap_samples=0)
        with self.assertRaises(ValueError):
            StatisticalValidationAnalyzer(confidence=1.0)
        with self.assertRaises(ValueError):
            StatisticalValidationAnalyzer().summarize([])


if __name__ == "__main__":
    unittest.main()
