import unittest
import numpy as np
import pandas as pd

from app.scoring.vfi_score import VFIScore


class TestAccumulationConfirmed(unittest.TestCase):
    def test_positive_vfi_alone_does_not_satisfy_accumulation_gate(self):
        index = pd.date_range("2024-01-05", periods=20, freq="W-FRI")
        data = pd.DataFrame({
            "VFI": np.full(20, 5.0),
            "VFI_Slope": np.full(20, 0.2),
            "VFI_Slope_4W": np.full(20, 0.2),
            "VFI_Slope_8W": np.full(20, 0.2),
            "VFI_Distance_Zero": np.full(20, 5.0),
            "VFI_Range_Position": np.full(20, 0.5),
            "VFI_Recovery_From_Min_Pct": np.full(20, 20.0),
            "VFI_State": ["BULLISH_FLOW"] * 20,
        }, index=index)
        result = VFIScore().calculate(data)
        self.assertFalse(bool(result.iloc[-1]["Accumulation_Confirmed"]))


if __name__ == "__main__":
    unittest.main()
