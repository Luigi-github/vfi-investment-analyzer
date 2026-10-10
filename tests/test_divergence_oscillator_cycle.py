import unittest

import numpy as np
import pandas as pd

from scripts.vfi_accumulation_divergence_study import add_divergence_cycle_features


class DivergenceOscillatorCycleTests(unittest.TestCase):
    def test_weekly_divergence_oscillator_has_pre_zero_and_cross_flags(self):
        idx = pd.date_range("2022-01-02", periods=130, freq="W-SUN")
        # Negative oscillator rising through zero near the end.
        osc = np.linspace(-10.0, 1.0, len(idx))
        frame = pd.DataFrame({"Volume_Divergence_MM": osc}, index=idx)
        result = add_divergence_cycle_features(
            frame, periods=(1, 2, 4), suffix="W", scale_window=52, min_periods=20
        )
        required = {
            "Div_Osc_Z_W",
            "Div_Osc_Slope_1W",
            "Div_Osc_Slope_2W_Normalized",
            "Div_Osc_Slope_4W_Normalized",
            "Div_Osc_Negative_Near_Zero_W",
            "Div_Osc_Strong_Rising_Slope_W",
            "Div_Osc_Early_Setup_W",
            "Div_Osc_Zero_Cross_Up_W",
        }
        self.assertTrue(required.issubset(result.columns))
        self.assertTrue(bool(result["Div_Osc_Zero_Cross_Up_W"].iloc[-1]))

    def test_oscillator_setup_is_not_true_above_zero(self):
        idx = pd.date_range("2022-01-02", periods=130, freq="W-SUN")
        osc = np.linspace(1.0, 10.0, len(idx))
        frame = pd.DataFrame({"Volume_Divergence_MM": osc}, index=idx)
        result = add_divergence_cycle_features(frame)
        self.assertFalse(bool(result["Div_Osc_Early_Setup_W"].iloc[-1]))


if __name__ == "__main__":
    unittest.main()
