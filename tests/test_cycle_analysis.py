import unittest

import pandas as pd

from app.backtest.cycle_analysis import VFICycleBacktestAnalyzer


def make_data():
    index = pd.date_range("2024-01-05", periods=14, freq="W-FRI")
    vfi = [-3, -2, -1, 0.5, 2, 3, 2, 1, 0, -1, -2, -1, 0.5, 2]
    stages = [
        "EARLY_RECOVERY", "SUSTAINED_RECOVERY", "SUSTAINED_RECOVERY",
        "ZERO_CROSS_UP", "ACCUMULATION", "BULLISH_FLOW", "WEAKENING",
        "ZERO_CROSS_DOWN", "BEARISH_FLOW", "EARLY_RECOVERY",
        "SUSTAINED_RECOVERY", "APPROACHING_ZERO", "ZERO_CROSS_UP",
        "ACCUMULATION",
    ]
    return pd.DataFrame(
        {
            "Close": [100, 102, 105, 110, 115, 120, 118, 112, 105, 103, 106, 109, 114, 118],
            "VFI": vfi,
            "VFI_Slope": [0, 1, 1, 1.5, 1.5, 1, -1, -1, -1, 1, 1, 1, 1.5, 1],
            "VFI_Slope_4W": [0, 0.2, 0.5, 0.8, 1, 1, 0.5, -0.5, -1, -0.5, 0.2, 0.5, 0.8, 1],
            "VFI_Slope_8W": [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.4, -0.2, -0.6, -0.3, 0.1, 0.3, 0.5, 0.6],
            "VFI_Cycle_Stage": stages,
        },
        index=index,
    )


class TestVFICycleBacktestAnalyzer(unittest.TestCase):

    def test_extract_cycles_creates_complete_and_incomplete_cycles(self):
        cycles = VFICycleBacktestAnalyzer().extract_cycles(make_data())
        self.assertEqual(len(cycles), 3)
        self.assertEqual(cycles.iloc[0]["Cycle_Type"], "BULLISH_CYCLE")
        self.assertTrue(bool(cycles.iloc[0]["Complete"]))
        self.assertFalse(bool(cycles.iloc[-1]["Complete"]))

    def test_bullish_cycle_quality_uses_only_cycle_information(self):
        cycles = VFICycleBacktestAnalyzer().extract_cycles(make_data())
        bullish = cycles[cycles["Cycle_Type"] == "BULLISH_CYCLE"].iloc[0]
        self.assertGreaterEqual(bullish["Weeks_SUSTAINED_RECOVERY"], 1)
        self.assertGreater(bullish["Peak_Return_From_Start_Pct"], 0)
        self.assertIn(
            bullish["Cycle_Quality"],
            {"EXCEPTIONAL", "STRONG", "MODERATE"},
        )

    def test_stage_entry_is_causal_and_only_counts_stage_start(self):
        events = VFICycleBacktestAnalyzer().stage_entry_analysis(
            make_data(),
            stages=("EARLY_RECOVERY", "SUSTAINED_RECOVERY", "ACCUMULATION"),
        )
        self.assertEqual(
            list(events["Cycle_Stage_Event"]),
            [
                "EARLY_RECOVERY", "SUSTAINED_RECOVERY", "ACCUMULATION",
                "EARLY_RECOVERY", "SUSTAINED_RECOVERY", "ACCUMULATION",
            ],
        )
        self.assertTrue(bool(events["Causal_Event"].all()))
        self.assertGreater(events.iloc[1]["VFI_Slope_4W"], 0)

    def test_stage_entries_do_not_change_when_future_rows_are_appended(self):
        analyzer = VFICycleBacktestAnalyzer()
        original = analyzer.stage_entry_analysis(make_data())
        extended = make_data().copy()

        future_index = pd.date_range(
            extended.index[-1] + pd.Timedelta(weeks=1),
            periods=4,
            freq="W-FRI",
        )
        future = extended.iloc[-1:].copy().loc[
            future_index.repeat(1)
        ]
        future.index = future_index
        future["Close"] = [130, 140, 150, 160]
        future["VFI"] = [2.5, 3, 3.5, 4]
        future["VFI_Slope"] = [0.5] * 4
        future["VFI_Slope_4W"] = [0.5] * 4
        future["VFI_Slope_8W"] = [0.5] * 4
        future["VFI_Cycle_Stage"] = ["BULLISH_FLOW"] * 4

        extended = pd.concat([extended, future])
        extended_events = analyzer.stage_entry_analysis(extended)

        self.assertEqual(
            list(original.index),
            list(extended_events.loc[original.index].index),
        )
        self.assertEqual(
            list(original["Cycle_Stage_Event"]),
            list(extended_events.loc[original.index]["Cycle_Stage_Event"]),
        )

    def test_validation_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            VFICycleBacktestAnalyzer().extract_cycles(
                pd.DataFrame({"Close": [1, 2]})
            )


if __name__ == "__main__":
    unittest.main()
