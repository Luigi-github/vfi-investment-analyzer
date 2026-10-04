import unittest

import numpy as np

from app.backtest.statistical_robustness import StatisticalRobustnessAnalyzer


class TestStatisticalRobustnessAnalyzer(unittest.TestCase):
    def test_bh_adjustment_preserves_original_order(self):
        analyzer = StatisticalRobustnessAnalyzer()
        result = analyzer.benjamini_hochberg([0.01, 0.20, 0.03])
        self.assertAlmostEqual(result[0], 0.03)
        self.assertAlmostEqual(result[2], 0.045)

    def test_bootstrap_excess_is_reproducible(self):
        analyzer = StatisticalRobustnessAnalyzer(bootstrap_samples=200, random_seed=7)
        first = analyzer.bootstrap_excess([10, 20, 30], [5, 10, 15])
        second = analyzer.bootstrap_excess([10, 20, 30], [5, 10, 15])
        self.assertEqual(first, second)
        self.assertEqual(first["Excess_Mean"], 10.0)

    def test_condition_testing_adjusts_multiple_p_values(self):
        analyzer = StatisticalRobustnessAnalyzer(bootstrap_samples=200, random_seed=7)
        result = analyzer.test_conditions({
            "strong": np.array([10, 12, 14, 16, 18]),
            "weak": np.array([-1, 0, 1, 0, -1]),
        })
        self.assertEqual(len(result), 2)
        self.assertIn("BH_Q_Value", result.columns)

    def test_stability_score_is_bounded(self):
        analyzer = StatisticalRobustnessAnalyzer()
        score = analyzer.stability_score(15, 70, 8, -10, benchmark_excess=8, significant=True)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            StatisticalRobustnessAnalyzer.benjamini_hochberg([0.1, 1.2])


if __name__ == "__main__":
    unittest.main()
