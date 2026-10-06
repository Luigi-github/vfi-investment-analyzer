import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.indicators.vfi import VFIIndicator


FIXTURE = Path(__file__).parent / "fixtures" / "vfi_conformance.csv"


class TestVFIConformance(unittest.TestCase):
    """Reference-data harness for an external TradingView export.

    The fixture is intentionally optional because TradingView export data is
    external to the repository. When supplied, the test compares the Python
    VFI against the frozen reference with a strict tolerance.
    """

    def test_external_reference_if_present(self):
        if not FIXTURE.exists():
            self.skipTest("TradingView conformance fixture not supplied.")
        reference = pd.read_csv(FIXTURE, parse_dates=["Date"]).set_index("Date")
        required = {"Open", "High", "Low", "Close", "Volume", "VFI_Reference"}
        self.assertTrue(required.issubset(reference.columns))
        calculated = VFIIndicator().calculate(reference)
        joined = calculated[["VFI"]].join(reference[["VFI_Reference"]], how="inner").dropna()
        self.assertGreater(len(joined), 0)
        error = np.abs(joined["VFI"] - joined["VFI_Reference"])
        self.assertLessEqual(float(error.max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
