import unittest
import pandas as pd

from app.research.multi_timeframe_crypto import MultiTimeframeCryptoAnalyzer


class TestMTFNoLookahead(unittest.TestCase):
    def _frame(self, index, value, stage="BULLISH_FLOW"):
        return pd.DataFrame(
            {
                "VFI": [value] * len(index),
                "VFI_Cycle_Stage": [stage] * len(index),
                "VFI_Slope": [1.0] * len(index),
                "VFI_Slope_4B": [1.0] * len(index),
                "VFI_Slope_8B": [1.0] * len(index),
            },
            index=pd.DatetimeIndex(pd.to_datetime(index, utc=True), name="Date"),
        )

    def test_higher_timeframe_signal_is_hidden_until_close(self):
        analyzer = MultiTimeframeCryptoAnalyzer()
        base_idx = pd.date_range("2026-01-05 00:00", periods=32, freq="1h", tz="UTC")
        calculated = {
            "1H": pd.DataFrame(
                {
                    "Close": [100.0] * len(base_idx),
                    "VFI": [0.0] * len(base_idx),
                    "VFI_Cycle_Stage": ["NEUTRAL"] * len(base_idx),
                },
                index=pd.DatetimeIndex(base_idx, name="Date"),
            ),
            "4H": self._frame(["2026-01-05 00:00", "2026-01-05 04:00"], 10.0),
            "1D": self._frame(["2026-01-05 00:00"], 20.0),
            "1W": self._frame(["2026-01-05 00:00"], 30.0),
        }

        out = analyzer.build_1h_confluence_history(calculated)

        self.assertTrue(pd.isna(out.loc[pd.Timestamp("2026-01-05 03:00", tz="UTC"), "VFI_4H"]))
        self.assertEqual(
            out.loc[pd.Timestamp("2026-01-05 04:00", tz="UTC"), "VFI_4H"], 10.0
        )

        # The daily candle opened at midnight but cannot affect hourly
        # decisions until the next midnight.
        self.assertTrue(pd.isna(out.loc[pd.Timestamp("2026-01-05 23:00", tz="UTC"), "VFI_1D"]))

    def test_1h_signal_is_not_backdated(self):
        analyzer = MultiTimeframeCryptoAnalyzer()
        idx = pd.date_range("2026-01-05 00:00", periods=3, freq="1h")
        calculated = {
            "1H": pd.DataFrame(
                {
                    "Close": [100.0, 101.0, 102.0],
                    "VFI": [1.0, 2.0, 3.0],
                    "VFI_Cycle_Stage": ["BULLISH_FLOW"] * 3,
                },
                index=pd.DatetimeIndex(idx, name="Date"),
            )
        }
        out = analyzer.build_1h_confluence_history(calculated)
        self.assertEqual(out.index[0], pd.Timestamp("2026-01-05 00:00"))
        self.assertEqual(out.loc[pd.Timestamp("2026-01-05 01:00"), "VFI_1H"], 2.0)


    def test_default_confluence_uses_latest_available_timeframe_timestamp(self):
        analyzer = MultiTimeframeCryptoAnalyzer()
        calculated = {
            "1W": self._frame(["2026-01-05 00:00"], 10.0),
            "1D": self._frame(["2026-01-09 00:00", "2026-01-10 00:00"], 20.0),
            "4H": self._frame(["2026-01-10 04:00", "2026-01-10 08:00"], 30.0),
            "1H": self._frame(["2026-01-10 11:00", "2026-01-10 12:00"], 40.0),
        }

        result = analyzer.confluence_at(calculated)

        self.assertEqual(result["Timestamp"], pd.Timestamp("2026-01-10 12:00", tz="UTC"))
        alignment = {row["Timeframe"]: row for row in result["Alignment"]}
        self.assertEqual(alignment["1H"]["Timestamp"], pd.Timestamp("2026-01-10 12:00", tz="UTC"))
        self.assertEqual(alignment["1D"]["Timestamp"], pd.Timestamp("2026-01-10 00:00", tz="UTC"))
        self.assertEqual(alignment["4H"]["Timestamp"], pd.Timestamp("2026-01-10 08:00", tz="UTC"))


if __name__ == "__main__":
    unittest.main()
