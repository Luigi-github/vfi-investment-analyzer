import unittest

import numpy as np
import pandas as pd

from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM
from scripts.vfi_accumulation_divergence_study import (
    MIN_WEEKLY_BARS_FOR_DIVERGENCE,
    add_divergence_cycle_features,
    add_vfi_features,
    build_asset_events,
    fresh_signal,
    to_weekly,
)


# Regression coverage for complete weekly oscillator history and zero-approach cycles.
class VfiDailyWeeklyStudyTests(unittest.TestCase):
    def test_fresh_signal_expires_in_daily_bars(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="D")
        signals = pd.Series([False, False, True, False, False, False, False, False, False, False], index=idx)
        result = fresh_signal(signals, 3)
        self.assertEqual(result.tolist(), [False, False, True, True, True, True, False, False, False, False])

    def test_weekly_resample_drops_incomplete_week(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="D")
        close = np.arange(10, dtype=float) + 10
        frame = pd.DataFrame({
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": np.ones(10),
        }, index=idx)
        weekly = to_weekly(frame, crypto=True)
        self.assertTrue((weekly.index.normalize() <= idx[-1].normalize()).all())
        self.assertTrue(all(weekly.index.dayofweek == 6))

    def test_weekly_divergence_zscore_waits_for_52_valid_values(self):
        idx = pd.date_range("2024-01-07", periods=120, freq="W-SUN")
        close = 100 + np.sin(np.arange(120) / 4.0) * 5 + np.arange(120) * 0.1
        frame = pd.DataFrame({
            "Open": close - 0.4,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1000 + (np.arange(120) % 11) * 20,
        }, index=idx)
        oscillator = VolumeDivergenceMM().calculate(frame)
        result = add_divergence_cycle_features(
            oscillator, periods=(1, 2, 4), suffix="W", scale_window=52, min_periods=52
        )
        self.assertTrue(result["Div_Osc_Z_W"].isna().all())

    def test_100_week_sample_is_not_mislabeled_as_a_full_divergence_cycle(self):
        self.assertEqual(MIN_WEEKLY_BARS_FOR_DIVERGENCE, 129)
        idx = pd.date_range("2024-01-01", periods=700, freq="D")
        close = 100 + np.arange(700) * 0.04 + np.sin(np.arange(700) / 6.0) * 2
        frame = pd.DataFrame({
            "Open": close - 0.2,
            "High": close + 0.8,
            "Low": close - 0.8,
            "Close": close,
            "Volume": 1000 + (np.arange(700) % 17) * 35,
        }, index=idx)
        _, result = build_asset_events(
            "TEST", frame, True, VFIIndicator(), VolumeDivergenceMM()
        )
        self.assertLess(result["Weekly_Bars"], MIN_WEEKLY_BARS_FOR_DIVERGENCE)
        self.assertLess(result["Weekly_Divergence_Valid_Bars"], 52)
        self.assertFalse(result["Weekly_Divergence_History_Sufficient"])
        self.assertEqual(result["Divergence_Cycle_Stage"], "NO_HISTORY")
        self.assertEqual(result["Weekly_Div_Signal"], "INSUFFICIENT_WEEKLY_HISTORY")

    def test_daily_vfi_features_expose_normalized_multiday_slopes(self):
        idx = pd.date_range("2024-01-01", periods=420, freq="D")
        close = 100 + np.arange(420) * 0.08 + np.sin(np.arange(420) / 7.0) * 2
        open_ = close - 0.2
        frame = pd.DataFrame({
            "Open": open_,
            "High": np.maximum(open_, close) + 1,
            "Low": np.minimum(open_, close) - 1,
            "Close": close,
            "Volume": 1000 + (np.arange(420) % 17) * 50,
        }, index=idx)
        result = add_vfi_features(frame, VFIIndicator())
        required = {
            "VFI_Z_252", "VFI_Slope_5D", "VFI_Slope_10D",
            "VFI_Slope_20D", "VFI_Strong_Rising_Slope", "VFI_Early_Setup",
            "VFI_Estimated_Days_To_Zero", "VFI_Zero_Cross_Up",
            "VFI_Zero_Cross_Up_Recent_Near_Zero",
        }
        self.assertTrue(required.issubset(result.columns))
        self.assertTrue(result["VFI"].notna().any())


if __name__ == "__main__":
    unittest.main()
