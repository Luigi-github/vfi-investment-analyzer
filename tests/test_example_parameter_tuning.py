import unittest

import pandas as pd

from scripts.tune_example_vfi_divergence_parameters import (
    divergence_setup,
    event_positions,
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

    def test_events_are_spaced_by_cooldown(self):
        index = pd.date_range("2025-01-01", periods=70, freq="D")
        mask = pd.Series(False, index=index)
        mask.iloc[[2, 10, 23, 24, 45]] = True
        self.assertEqual(event_positions(mask, cooldown=20), [2, 23, 45])


if __name__ == "__main__":
    unittest.main()
