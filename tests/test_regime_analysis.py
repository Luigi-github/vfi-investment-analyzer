import unittest

import pandas as pd

from app.backtest.regime_analysis import RegimeAnalyzer


class TestRegimeAnalyzer(unittest.TestCase):
    def test_classifies_bull_bear_and_sideways(self):
        dates = pd.date_range("2024-01-05", periods=30, freq="W-FRI")
        close = [100.0] * 26 + [120.0, 105.0, 89.0, 89.0]
        benchmark = pd.DataFrame({"Close": close}, index=dates)
        result = RegimeAnalyzer(lookback_weeks=26).classify(benchmark)
        self.assertEqual(result.iloc[26]["Market_Regime"], "BULL")
        self.assertEqual(result.iloc[28]["Market_Regime"], "SIDEWAYS")
        self.assertEqual(result.iloc[29]["Market_Regime"], "BEAR")

    def test_event_labeling_uses_historical_benchmark_state(self):
        dates = pd.date_range("2024-01-05", periods=30, freq="W-FRI")
        benchmark = pd.DataFrame({"Close": [100.0] * 26 + [120.0] * 4}, index=dates)
        events = pd.DataFrame({"Return_4W": [10.0]}, index=[dates[27]])
        labeled = RegimeAnalyzer(lookback_weeks=26).label_events(events, benchmark)
        self.assertEqual(labeled.iloc[0]["Market_Regime"], "BULL")

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            RegimeAnalyzer(lookback_weeks=0)
        with self.assertRaises(ValueError):
            RegimeAnalyzer(bull_threshold=-0.1, bear_threshold=0.0)


if __name__ == "__main__":
    unittest.main()
