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
