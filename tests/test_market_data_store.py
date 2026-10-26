import tempfile
import unittest

import pandas as pd

from app.data.market_data_store import ColombiaMarketDataStore, STORE_COLUMNS


def sample_frame():
    idx = pd.to_datetime(["2026-10-22", "2026-10-23"])
    return pd.DataFrame({
        "Open": [10.0, 11.0], "High": [12.0, 13.0], "Low": [9.0, 10.0],
        "Close": [11.0, 12.0], "Volume": [1000.0, 1200.0],
        "Adj Close": [11.0, 12.0], "Dividends": [0.0, 0.0],
        "Stock Splits": [0.0, 0.0],
    }, index=idx)


class TestColombiaMarketDataStore(unittest.TestCase):
    def test_valid_recent_data_passes(self):
        report = ColombiaMarketDataStore.validate(
            sample_frame(), as_of=pd.Timestamp("2026-10-25"), max_stale_calendar_days=7
        )
        self.assertTrue(report["Valid"])
        self.assertTrue(report["Fresh"])
        self.assertEqual(report["Status"], "OK")
        self.assertEqual(report["Last_Date"], "2026-10-23")

    def test_stale_data_is_not_marked_current(self):
        report = ColombiaMarketDataStore.validate(
            sample_frame(), as_of=pd.Timestamp("2026-11-10"), max_stale_calendar_days=7
        )
        self.assertTrue(report["Valid"])
        self.assertFalse(report["Fresh"])
        self.assertEqual(report["Status"], "STALE")
        self.assertIn("STALE_LAST_OBSERVATION", report["Issues"])

    def test_invalid_ohlc_is_rejected(self):
        frame = sample_frame()
        frame.loc[pd.Timestamp("2026-10-23"), "High"] = 9.0
        report = ColombiaMarketDataStore.validate(frame, as_of=pd.Timestamp("2026-10-25"))
        self.assertFalse(report["Valid"])
        self.assertEqual(report["Status"], "INVALID")
        self.assertIn("HIGH_BELOW_OHLC", report["Issues"])

    def test_empty_data_is_not_current(self):
        report = ColombiaMarketDataStore.validate(pd.DataFrame())
        self.assertFalse(report["Valid"])
        self.assertFalse(report["Fresh"])
        self.assertEqual(report["Status"], "NO_DATA")

    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ColombiaMarketDataStore(tmp)
            sample_frame().to_csv(store.cache_path("ISA.CL"), index_label="Date", date_format="%Y-%m-%d")
            loaded = store.load_cached("ISA.CL")
            self.assertEqual(len(loaded), 2)
            self.assertListEqual(list(loaded.columns), STORE_COLUMNS)
            self.assertEqual(float(loaded.iloc[-1]["Close"]), 12.0)


if __name__ == "__main__":
    unittest.main()
