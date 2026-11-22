import unittest
from unittest.mock import Mock, patch

import pandas as pd

from scripts import run_colombia_three_rankings as pipeline


class YahooAdjustmentPolicyTests(unittest.TestCase):
    def test_daily_download_uses_unadjusted_ohlc(self):
        frame = pd.DataFrame(
            {
                "Open": [10.0, 11.0],
                "High": [11.0, 12.0],
                "Low": [9.0, 10.0],
                "Close": [10.5, 11.5],
                "Volume": [1000.0, 1200.0],
            },
            index=pd.to_datetime(["2026-11-18", "2026-11-19"]),
        )
        ticker = Mock()
        ticker.history.return_value = frame
        with patch.object(pipeline.yf, "Ticker", return_value=ticker):
            result = pipeline.download_daily("CIBEST.CL", period="5d")
        self.assertFalse(result.empty)
        self.assertIs(ticker.history.call_args.kwargs["auto_adjust"], False)

    def test_native_weekly_download_uses_unadjusted_ohlc(self):
        frame = pd.DataFrame(
            {
                "Open": [10.0],
                "High": [12.0],
                "Low": [9.0],
                "Close": [11.0],
                "Volume": [5000.0],
            },
            index=pd.to_datetime(["2026-11-16"]),
        )
        ticker = Mock()
        ticker.history.return_value = frame
        with patch.object(pipeline.yf, "Ticker", return_value=ticker):
            result = pipeline.download_native_weekly("PFCIBEST.CL")
        self.assertFalse(result.empty)
        self.assertIs(ticker.history.call_args.kwargs["auto_adjust"], False)


if __name__ == "__main__":
    unittest.main()
