import unittest

import pandas as pd

from app.backtest.condition_discovery import ConditionDiscoveryAnalyzer


class TestConditionDiscovery(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range("2020-01-03", periods=8, freq="W-FRI")
        self.events = pd.DataFrame(
            {
                "VFI": [-8, -7, -6, -5, -4, -3, -2, -1],
                "VFI_Slope": [1, 1, 1, 1, -1, -1, -1, -1],
                "VFI_Slope_4W": [1, 1, 1, 1, -1, -1, -1, -1],
                "VFI_Slope_8W": [1, 1, 1, 1, -1, -1, -1, -1],
                "VFI_Distance_Zero": [8, 7, 6, 5, 4, 3, 2, 1],
                "VFI_Range_Position": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "Return_2W": [30, 25, 20, 15, -5, -10, -15, -20],
            },
            index=dates,
        )
        self.conditions = {
            "Deep VFI + positive slope": (
                (self.events["VFI"] < -5)
                & (self.events["VFI_Slope"] > 0)
            ),
            "Positive slope": self.events["VFI_Slope"] > 0,
        }

    def test_standard_conditions_match_existing_vfi_hypotheses(self):
        analyzer = ConditionDiscoveryAnalyzer(primary_horizon=2)
        conditions = analyzer.build_standard_conditions(self.events)

        self.assertIn("VFI < 0", conditions)
        self.assertIn("VFI < -5 AND Slope > 0", conditions)
        self.assertIn(
            "VFI < 0 AND Slope > 0 AND Slope 8W > 0",
            conditions,
        )
        self.assertEqual(len(conditions), 12)


    def test_discovery_ranks_using_train_only(self):
        analyzer = ConditionDiscoveryAnalyzer(
            primary_horizon=2,
            minimum_events=2,
        )
        train = self.events.iloc[:4]
        discovered = analyzer.discover(train, {
            name: mask.loc[train.index]
            for name, mask in self.conditions.items()
        })

        self.assertFalse(discovered.empty)
        self.assertEqual(
            discovered.iloc[0]["Condition"],
            "Deep VFI + positive slope",
        )

    def test_walk_forward_discovery_selects_on_train_and_evaluates_test(self):
        analyzer = ConditionDiscoveryAnalyzer(
            primary_horizon=2,
            minimum_events=2,
        )
        windows = analyzer.walk_forward_discovery(
            self.events,
            initial_train_weeks=4,
            test_weeks=2,
            step_weeks=2,
            condition_builder=lambda frame: {
                name: mask.loc[frame.index]
                for name, mask in self.conditions.items()
            },
        )

        self.assertGreaterEqual(len(windows), 1)
        first = windows[0]
        self.assertEqual(
            first["selected"].iloc[0]["Condition"],
            "Deep VFI + positive slope",
        )
        self.assertEqual(
            first["test_results"].iloc[0]["Condition"],
            "Deep VFI + positive slope",
        )
        self.assertEqual(first["test_results"].iloc[0]["Count"], 0)

    def test_walk_forward_requires_datetime_index(self):
        analyzer = ConditionDiscoveryAnalyzer(primary_horizon=2)

        with self.assertRaises(ValueError):
            analyzer.walk_forward_discovery(
                self.events.reset_index(drop=True),
                initial_train_weeks=2,
                test_weeks=2,
                step_weeks=2,
            )

    def test_cost_adjusted_returns_are_reported(self):
        analyzer = ConditionDiscoveryAnalyzer(
            primary_horizon=2,
            minimum_events=2,
            fee_bps=10,
            slippage_bps=10,
        )
        result = analyzer.evaluate_conditions(
            self.events,
            {"all": pd.Series(True, index=self.events.index)},
        )
        self.assertIn("Average_Net_Return", result.columns)
        self.assertLess(
            result.iloc[0]["Average_Net_Return"],
            result.iloc[0]["Average_Return"],
        )

    def test_cost_adjusted_walk_forward_preserves_gross_and_net_metrics(self):
        analyzer = ConditionDiscoveryAnalyzer(
            primary_horizon=2,
            minimum_events=2,
            fee_bps=10,
            slippage_bps=10,
        )
        windows = analyzer.walk_forward_discovery(
            self.events,
            initial_train_weeks=4,
            test_weeks=2,
            step_weeks=2,
            condition_builder=lambda frame: {
                "Positive slope": frame["VFI_Slope"] > 0,
            },
        )
        summary = analyzer.summarize_walk_forward(windows)
        self.assertIn("Train_Average_Net_Return", summary.columns)
        self.assertIn("Test_Average_Net_Return", summary.columns)

    def test_selected_conditions_are_evaluated_on_test_without_reranking(self):
        analyzer = ConditionDiscoveryAnalyzer(
            primary_horizon=2,
            minimum_events=2,
        )
        train = self.events.iloc[:4]
        test = self.events.iloc[4:]

        train_conditions = {
            name: mask.loc[train.index]
            for name, mask in self.conditions.items()
        }
        test_conditions = {
            name: mask.loc[test.index]
            for name, mask in self.conditions.items()
        }

        discovered = analyzer.discover(train, train_conditions)
        selected_test = analyzer.evaluate_selected(
            test,
            discovered.head(1),
            test_conditions,
        )

        self.assertEqual(
            selected_test.iloc[0]["Condition"],
            "Deep VFI + positive slope",
        )
        self.assertEqual(selected_test.iloc[0]["Count"], 0)


if __name__ == "__main__":
    unittest.main()
