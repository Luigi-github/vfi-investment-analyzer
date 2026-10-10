import unittest

import pandas as pd

from app.research.multi_timeframe_crypto import MultiTimeframeCryptoAnalyzer


def frame(timestamp, stage):
    idx = pd.DatetimeIndex([pd.Timestamp(timestamp, tz="UTC")], name="Date")
    return pd.DataFrame(
        {
            "Close": [100.0],
            "VFI": [1.0],
            "VFI_Cycle_Stage": [stage],
            "VFI_Slope": [0.1],
            "VFI_Slope_4B": [0.1],
            "VFI_Slope_8B": [0.1],
        },
        index=idx,
    )


class MissingTimeframeScoreTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = MultiTimeframeCryptoAnalyzer()

    def test_single_bullish_hourly_frame_cannot_score_100(self):
        calculated = {"1H": frame("2026-10-10 12:00:00", "BULLISH_FLOW")}
        result = self.analyzer.confluence_at(calculated)
        self.assertAlmostEqual(result["Score"], 57.5)
        self.assertEqual(result["Direction"], "MIXED")

    def test_score_uses_fixed_weights_when_daily_and_intraday_are_available(self):
        calculated = {
            "1D": frame("2026-10-09 00:00:00", "BULLISH_FLOW"),
            "4H": frame("2026-10-10 08:00:00", "BULLISH_FLOW"),
            "1H": frame("2026-10-10 12:00:00", "BULLISH_FLOW"),
        }
        result = self.analyzer.confluence_at(calculated)
        # 30 + 25 + 15 bullish points out of the fixed 100-point scale.
        self.assertAlmostEqual(result["Score"], 85.0)

    def test_history_score_does_not_renormalize_missing_weekly_frame(self):
        calculated = {
            "1H": frame("2026-10-10 12:00:00", "BULLISH_FLOW"),
            "4H": frame("2026-10-10 08:00:00", "BULLISH_FLOW"),
            "1D": frame("2026-10-09 00:00:00", "BULLISH_FLOW"),
        }
        history = self.analyzer.build_1h_confluence_history(calculated)
        self.assertFalse(history.empty)
        self.assertAlmostEqual(float(history["MTF_Score"].iloc[-1]), 85.0)


if __name__ == "__main__":
    unittest.main()
