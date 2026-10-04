import unittest

import numpy as np
import pandas as pd

from app.indicators.vfi import VFIIndicator
from app.analysis.vfi_analyzer import VFIAnalyzer
from app.scoring.vfi_score import VFIScore


class TestPipelineNoLookahead(unittest.TestCase):

    def make_ohlcv(self, rows):
        index = pd.date_range(
            "2018-01-05",
            periods=rows,
            freq="W-FRI",
        )

        close = (
            100.0
            + np.arange(rows) * 0.15
            + 3.0 * np.sin(np.arange(rows) / 9.0)
        )

        open_price = close - 0.5
        high = close + 1.0
        low = close - 1.0

        volume = (
            1_000_000
            + (np.arange(rows) % 17) * 25_000
        )

        return pd.DataFrame(
            {
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=index,
        )

    def build_pipeline(self, data):
        indicator = VFIIndicator()
        analyzer = VFIAnalyzer()
        scorer = VFIScore()

        vfi_data = indicator.calculate(data)
        analyzed = analyzer.analyze(vfi_data)
        return scorer.calculate(analyzed)

    def test_future_rows_do_not_change_historical_pipeline_features(self):
        initial = self.make_ohlcv(285)
        extended = self.make_ohlcv(300)

        initial_result = self.build_pipeline(initial)
        extended_result = self.build_pipeline(extended)

        comparison_index = initial.index[260:]

        columns = [
            "VFI",
            "VFI_Signal",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Rolling_Min",
            "VFI_Rolling_Max",
            "VFI_Range_Position",
            "VFI_Recovery_From_Min_Pct",
            "VFI_State",
            "VFI_Recovery_Score",
            "Accumulation_Score",
            "Accumulation_Confirmed",
            "VFI_Cycle_Stage",
        ]

        for column in columns:
            if (
                column not in initial_result.columns
                or column not in extended_result.columns
            ):
                continue

            if pd.api.types.is_numeric_dtype(
                initial_result[column]
            ):
                np.testing.assert_allclose(
                    initial_result.loc[
                        comparison_index, column
                    ].to_numpy(),
                    extended_result.loc[
                        comparison_index, column
                    ].to_numpy(),
                    equal_nan=True,
                    err_msg=(
                        f"Historical values changed for {column} "
                        "after future rows were appended."
                    ),
                )
            else:
                pd.testing.assert_series_equal(
                    initial_result.loc[
                        comparison_index, column
                    ],
                    extended_result.loc[
                        comparison_index, column
                    ],
                    check_names=False,
                )


    def test_recovery_signal_analyzer_uses_configured_primary_horizon(self):
        from app.backtest.recovery_signal_analysis import RecoverySignalAnalyzer

        dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
        data = pd.DataFrame(
            {
                "Close": [100.0, 110.0, 121.0],
                "VFI": [-3.0, -2.0, -1.0],
                "VFI_State": ["BEARISH_FLOW"] * 3,
                "VFI_Slope": [0.1, 0.1, 0.1],
                "VFI_Slope_4W": [0.1, 0.1, 0.1],
                "VFI_Slope_8W": [0.1, 0.1, 0.1],
                "VFI_Distance_Zero": [3.0, 2.0, 1.0],
                "VFI_Range_Position": [0.1, 0.2, 0.3],
                "VFI_Cycle_Stage": ["EARLY_RECOVERY"] * 3,
            },
            index=dates,
        )

        analyzer = RecoverySignalAnalyzer(
            horizons=(1, 2),
            primary_horizon=1,
        )
        events = analyzer.build_event_dataset(data)

        self.assertAlmostEqual(events.iloc[0]["Primary_Return"], 10.0)
        self.assertEqual(events.iloc[0]["Primary_Result"], "POSITIVE")

    def test_recovery_signal_analyzer_rejects_primary_horizon_not_in_horizons(self):
        from app.backtest.recovery_signal_analysis import RecoverySignalAnalyzer

        with self.assertRaises(ValueError):
            RecoverySignalAnalyzer(horizons=(4, 8), primary_horizon=26)


if __name__ == "__main__":
    unittest.main()
