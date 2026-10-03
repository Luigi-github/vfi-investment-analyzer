import unittest

import numpy as np
import pandas as pd

from app.backtest.forward_risk import ForwardRiskAnalyzer


class TestForwardRisk(unittest.TestCase):

    def test_calculates_forward_excursions_and_drawdown(self):
        dates = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
        data = pd.DataFrame(
            {"Close": [100.0, 110.0, 90.0, 120.0, 80.0]},
            index=dates,
        )

        result = ForwardRiskAnalyzer(horizons=(2,)).calculate(data)

        self.assertAlmostEqual(
            result.loc[dates[0], "Max_Favorable_2W"],
            20.0,
        )
        self.assertAlmostEqual(
            result.loc[dates[0], "Max_Adverse_2W"],
            -10.0,
        )
        self.assertAlmostEqual(
            result.loc[dates[0], "Max_Drawdown_2W"],
            -18.18181818,
            places=5,
        )

    def test_does_not_use_partial_future_windows(self):
        dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
        data = pd.DataFrame(
            {"Close": [100.0, 110.0, 90.0]},
            index=dates,
        )

        result = ForwardRiskAnalyzer(horizons=(2,)).calculate(data)

        self.assertFalse(np.isnan(result.loc[dates[0], "Max_Drawdown_2W"]))
        self.assertTrue(np.isnan(result.loc[dates[1], "Max_Drawdown_2W"]))

    def test_future_rows_do_not_change_completed_historical_window(self):
        initial_dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
        extended_dates = pd.date_range("2024-01-05", periods=6, freq="W-FRI")

        initial = pd.DataFrame(
            {"Close": [100.0, 110.0, 90.0, 120.0]},
            index=initial_dates,
        )
        extended = pd.DataFrame(
            {"Close": [100.0, 110.0, 90.0, 120.0, 80.0, 130.0]},
            index=extended_dates,
        )

        analyzer = ForwardRiskAnalyzer(horizons=(2,))
        initial_result = analyzer.calculate(initial)
        extended_result = analyzer.calculate(extended)

        for column in (
            "Max_Favorable_2W",
            "Max_Adverse_2W",
            "Max_Drawdown_2W",
        ):
            self.assertAlmostEqual(
                initial_result.loc[initial_dates[0], column],
                extended_result.loc[extended_dates[0], column],
            )


if __name__ == "__main__":
    unittest.main()
