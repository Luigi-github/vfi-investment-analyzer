import unittest

import pandas as pd

from app.backtest.recovery_cluster_analyzer import (
    RecoveryClusterAnalyzer,
)
from app.backtest.recovery_event_clustering import (
    RecoveryEventClusterer,
)


class TestBacktestNoLookahead(unittest.TestCase):

    def setUp(self):
        self.analyzer = RecoveryClusterAnalyzer(
            horizons=(1,),
            primary_horizon=1,
        )
        self.clusterer = RecoveryEventClusterer(
            minimum_separation_weeks=8,
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

    def test_causal_cluster_ids_are_stable_when_future_events_are_added(self):
        dates = pd.to_datetime(
            ["2024-01-05", "2024-02-02", "2024-04-05"]
        )

        initial = pd.DataFrame(
            {
                "VFI": [-2.0, -1.0],
                "VFI_Cycle_Stage": ["EARLY_RECOVERY", "EARLY_RECOVERY"],
                "Close": [100.0, 101.0],
            },
            index=dates[:2],
        )

        extended = pd.concat(
            [
                initial,
                pd.DataFrame(
                    {
                        "VFI": [-3.0],
                        "VFI_Cycle_Stage": ["EARLY_RECOVERY"],
                        "Close": [102.0],
                    },
                    index=dates[2:],
                ),
            ]
        )

        initial_clustered = self.clusterer.cluster_events_causal(initial)
        extended_clustered = self.clusterer.cluster_events_causal(extended)

        pd.testing.assert_series_equal(
            initial_clustered["Recovery_Cluster_ID"],
            extended_clustered.loc[
                initial.index,
                "Recovery_Cluster_ID",
            ],
            check_names=False,
        )

    def test_causal_cluster_representative_is_first_event(self):
        events = pd.DataFrame(
            {
                "VFI": [-2.0, -5.0],
                "VFI_Cycle_Stage": ["EARLY_RECOVERY", "EARLY_RECOVERY"],
                "Close": [100.0, 101.0],
            },
            index=pd.to_datetime(["2024-01-05", "2024-02-02"]),
        )

        clustered = self.clusterer.cluster_events_causal(events)
        representatives = self.clusterer.select_causal_representative_events(
            clustered
        )

        self.assertEqual(
            representatives.index[0],
            pd.Timestamp("2024-01-05"),
        )

        self.assertEqual(
            int(clustered.loc[
                pd.Timestamp("2024-01-05"),
                "Recovery_Events_Seen",
            ]),
            1,
        )

        self.assertEqual(
            int(clustered.loc[
                pd.Timestamp("2024-02-02"),
                "Recovery_Events_Seen",
            ]),
            2,
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
