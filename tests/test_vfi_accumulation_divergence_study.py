import unittest

import numpy as np
import pandas as pd

from app.indicators.vfi import VFIIndicator
from scripts.vfi_accumulation_divergence_study import (
    add_vfi_features,
    fresh_signal,
    to_weekly,
)


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
            "VFI_Zero_Cross_Up",
        }
        self.assertTrue(required.issubset(result.columns))
        self.assertTrue(result["VFI"].notna().any())


if __name__ == "__main__":
    unittest.main()
