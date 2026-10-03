import unittest

import pandas as pd

from app.backtest.recovery_cluster_analyzer import (
    RecoveryClusterAnalyzer,
)


class TestBacktestNoLookahead(unittest.TestCase):

    def setUp(self):
        self.analyzer = RecoveryClusterAnalyzer(
            horizons=(1,),
            primary_horizon=1,
        )

    def make_cluster(self, dates, vfi_values):
        return pd.DataFrame(
            {
                "Close": [100.0] * len(dates),
                "VFI": vfi_values,
                "VFI_Slope": [0.10] * len(dates),
                "VFI_Slope_8W": [0.10] * len(dates),
                "VFI_Range_Position": [0.50] * len(dates),
                "Recovery_Cluster_ID": [1] * len(dates),
            },
            index=pd.to_datetime(dates),
        )

    def test_forward_returns_at_past_dates_do_not_change_when_future_data_is_added(self):
        dates = [
            "2024-01-05",
            "2024-01-12",
            "2024-01-19",
        ]

        initial = self.make_cluster(
            dates[:2],
            [-2.0, -1.0],
        )

        extended = self.make_cluster(
            dates,
            [-2.0, -1.0, 0.0],
        )

        initial_returns = (
            self.analyzer
            .calculate_forward_returns(initial)
        )

        extended_returns = (
            self.analyzer
            .calculate_forward_returns(extended)
        )

        self.assertAlmostEqual(
            initial_returns.loc[
                pd.Timestamp("2024-01-05"),
                "Return_1W",
            ],
            extended_returns.loc[
                pd.Timestamp("2024-01-05"),
                "Return_1W",
            ],
        )

    def test_causal_representative_does_not_move_when_future_event_is_added(self):
        dates = [
            "2024-01-05",
            "2024-01-12",
            "2024-01-19",
        ]

        data_available_at_signal = self.make_cluster(
            dates[:2],
            [-2.0, -1.0],
        )

        data_with_future_event = self.make_cluster(
            dates,
            [-2.0, -1.0, -5.0],
        )

        representative_at_signal = (
            self.analyzer
            .select_representatives(
                data_available_at_signal,
                "CAUSAL_FIRST_RECOVERY",
            )
        )

        representative_with_future_data = (
            self.analyzer
            .select_representatives(
                data_with_future_event,
                "CAUSAL_FIRST_RECOVERY",
            )
        )

        signal_date = pd.Timestamp("2024-01-05")

        self.assertEqual(
            representative_at_signal.index[0],
            signal_date,
        )

        self.assertEqual(
            representative_with_future_data.index[0],
            signal_date,
            "A future VFI observation must not move a causal "
            "historical representative signal.",
        )


if __name__ == "__main__":
    unittest.main()
