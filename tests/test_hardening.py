import unittest
import numpy as np
import pandas as pd

from app.backtest.block_bootstrap import block_bootstrap_mean
from app.backtest.portfolio_backtest import PortfolioBacktester
from app.backtest.signal_outcome import SignalOutcomeBuilder
from app.features.vfi_normalization import add_vfi_normalized_features
from app.indicators.vfi import VFIIndicator
from app.research.reproducibility import dataframe_sha256


class TestHardening(unittest.TestCase):
    def test_vfi_is_invariant_to_future_rows(self):
        dates = pd.date_range("2020-01-03", periods=180, freq="W-FRI")
        base = pd.DataFrame({
            "Open": np.linspace(100, 120, len(dates)),
            "High": np.linspace(101, 121, len(dates)),
            "Low": np.linspace(99, 119, len(dates)),
            "Close": np.linspace(100.5, 120.5, len(dates)),
            "Volume": np.full(len(dates), 100000.0),
        }, index=dates)
        indicator = VFIIndicator()
        short = indicator.calculate(base.iloc[:150])
        long = indicator.calculate(base)
        pd.testing.assert_series_equal(
            short["VFI"].dropna(),
            long.loc[short.index, "VFI"].dropna(),
            check_names=False,
        )

    def test_normalized_vfi_is_scale_aware(self):
        dates = pd.date_range("2020-01-03", periods=130, freq="W-FRI")
        data = pd.DataFrame({
            "VFI": np.arange(130, dtype=float),
            "VFI_Rolling_Min": np.arange(130, dtype=float) - 10,
            "VFI_Rolling_Max": np.arange(130, dtype=float) + 10,
        }, index=dates)
        result = add_vfi_normalized_features(data, 52, 52)
        self.assertTrue(result["VFI_ZScore"].notna().iloc[-1])
        self.assertTrue(result["VFI_Percentile"].notna().iloc[-1])
        self.assertAlmostEqual(result["VFI_Normalized_Range"].iloc[-1], 0.5)

    def test_dataframe_hash_is_stable(self):
        frame = pd.DataFrame({"VFI": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
        self.assertEqual(dataframe_sha256(frame), dataframe_sha256(frame.copy()))

    def test_block_bootstrap_is_reproducible(self):
        result1 = block_bootstrap_mean([1, 2, 3, 4, 5], block_size=2, iterations=100, random_state=7)
        result2 = block_bootstrap_mean([1, 2, 3, 4, 5], block_size=2, iterations=100, random_state=7)
        self.assertEqual(result1, result2)

    def test_signal_outcome_enters_next_observation(self):
        dates = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
        prices = {"EC": pd.DataFrame({
            "Close": np.arange(100.0, 140.0),
            "Adj Close": np.arange(100.0, 140.0),
        }, index=dates)}
        signals = pd.DataFrame(
            {"Symbol": ["EC"], "Signal_Score": [80]},
            index=pd.DatetimeIndex(["2024-01-05"]),
        )
        result = SignalOutcomeBuilder(4).build(prices, signals)
        self.assertEqual(result.iloc[0]["Entry_Date"], dates[1])
        self.assertEqual(result.iloc[0]["Exit_Date"], dates[5])

    def test_portfolio_does_not_execute_on_signal_bar(self):
        dates = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
        prices = {"EC": pd.DataFrame({"Close": [100, 110, 111, 112, 113, 114, 115, 116]}, index=dates)}
        signals = pd.DataFrame({"Symbol": ["EC"], "Signal_Score": [90]}, index=pd.DatetimeIndex([dates[0]]))
        result = PortfolioBacktester(horizon_weeks=2).run(prices, signals)
        self.assertEqual(result["trades"].iloc[0]["Signal_Date"], dates[0])
        self.assertEqual(result["trades"].iloc[0]["Entry_Date"], dates[1])

    def test_signal_score_controls_position_selection(self):
        dates = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
        prices = {
            "A": pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates),
            "B": pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates),
        }
        signals = pd.DataFrame(
            {"Symbol": ["A", "B"], "Signal_Score": [50, 90]},
            index=pd.DatetimeIndex([dates[0], dates[0]]),
        )
        result = PortfolioBacktester(horizon_weeks=2, max_positions=1).run(prices, signals)
        self.assertEqual(result["trades"].iloc[0]["Symbol"], "B")


if __name__ == "__main__":
    unittest.main()
