import unittest

import numpy as np
import pandas as pd

from scripts.tune_example_vfi_divergence_parameters import (
    DIVERGENCE_CONFIGS,
    divergence_features,
    divergence_setup,
    event_positions,
    evaluate_mask,
    vfi_setup,
)


class ExampleParameterTuningTests(unittest.TestCase):
    def test_vfi_setup_requires_negative_near_zero_and_positive_slopes(self):
        frame = pd.DataFrame({
            "VFI": [-0.2, -0.2, 0.1],
            "VFI_Z": [-0.4, -0.4, 0.1],
            "SLOPE_5": [0.1, -0.1, 0.1],
            "SLOPE_10": [0.2, 0.2, 0.2],
            "SLOPE_20": [0.3, 0.3, 0.3],
            "SLOPE_10_Z": [0.3, 0.3, 0.3],
            "SLOPE_20_Z": [0.4, 0.4, 0.4],
            "DAYS_TO_ZERO": [5.0, 5.0, 5.0],
        })
        params = {
            "vfi_near_zero_z": 0.75,
            "vfi_slope_min": 0.2,
            "vfi_max_days_to_zero": 10,
        }
        result = vfi_setup(frame, params).tolist()
        self.assertEqual(result, [True, False, False])

    def test_divergence_setup_requires_negative_oscillator_and_strong_rising_slopes(self):
        frame = pd.DataFrame({
            "Volume_Divergence_MM": [-0.2, -0.2, 0.1],
            "DIV_Z": [-0.4, -0.4, 0.1],
            "DIV_SLOPE_1": [0.1, 0.1, 0.1],
            "DIV_SLOPE_2": [0.2, -0.1, 0.2],
            "DIV_SLOPE_4": [0.3, 0.3, 0.3],
            "DIV_SLOPE_2_Z": [0.3, 0.3, 0.3],
            "DIV_SLOPE_4_Z": [0.4, 0.4, 0.4],
            "WEEKS_TO_ZERO": [3.0, 3.0, 3.0],
        })
        params = {
            "div_near_zero_z": 0.75,
            "div_slope_min": 0.2,
            "div_max_weeks_to_zero": 8,
        }
        result = divergence_setup(frame, params).tolist()
        self.assertEqual(result, [True, False, False])

    def test_divergence_config_grid_includes_tradingview_control_and_variants(self):
        names = [config["name"] for config in DIVERGENCE_CONFIGS]
        self.assertEqual(names[0], "TV_DEFAULT")
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("FASTER_WMA", names)
        self.assertIn("SLOWER_WMA", names)
        self.assertIn("EARLIER_PIVOTS", names)
        self.assertIn("STRICTER_PIVOTS", names)
        self.assertEqual(
            {key: DIVERGENCE_CONFIGS[0][key] for key in ("vl1", "vl2", "pivot_left", "pivot_right", "range_lower", "range_upper")},
            {"vl1": 5, "vl2": 8, "pivot_left": 5, "pivot_right": 5, "range_lower": 5, "range_upper": 60},
        )

    def test_divergence_features_calculate_for_each_candidate_config(self):
        index = pd.date_range("2020-01-03", periods=220, freq="W-FRI")
        close = pd.Series(100 + 0.08 * range(len(index)) + 5 * np.sin(np.arange(len(index)) / 7), index=index)
        frame = pd.DataFrame({
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1000 + 200 * np.sin(np.arange(len(index)) / 3) ** 2,
        }, index=index)
        outputs = [divergence_features(frame, config) for config in DIVERGENCE_CONFIGS]
        self.assertTrue(all("Volume_Divergence_MM" in output.columns for output in outputs))
        self.assertTrue(all(len(output) == len(frame) for output in outputs))
        self.assertTrue(any(output["Volume_Divergence_MM"].notna().any() for output in outputs))

    def test_events_are_spaced_by_cooldown(self):
        index = pd.date_range("2025-01-01", periods=70, freq="D")
        mask = pd.Series(False, index=index)
        mask.iloc[[2, 10, 23, 24, 45]] = True
        self.assertEqual(event_positions(mask, cooldown=20), [2, 23, 45])

    def test_training_forward_returns_cannot_cross_holdout_boundary(self):
        index = pd.date_range("2020-01-01", periods=100, freq="D")
        close = pd.Series(range(100, 200), index=index, dtype=float)
        mask = pd.Series(False, index=index)
        mask.iloc[[20, 50, 75]] = True
        train, test = evaluate_mask(mask, "TEST", close, index[60])
        self.assertEqual([e["Date"] for e in train], [index[20].strftime("%Y-%m-%d"), index[50].strftime("%Y-%m-%d")])
        self.assertTrue(pd.notna(train[0]["R20"]))
        self.assertTrue(pd.isna(train[1]["R20"]))
        self.assertEqual([e["Date"] for e in test], [index[75].strftime("%Y-%m-%d")])
        self.assertTrue(pd.notna(test[0]["R20"]))


if __name__ == "__main__":
    unittest.main()
