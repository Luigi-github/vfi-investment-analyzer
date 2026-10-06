import unittest
import pandas as pd
import numpy as np
from app.research.multi_timeframe_crypto import MultiTimeframeCryptoAnalyzer

class TestMultiTimeframeCrypto(unittest.TestCase):
    def test_forward_returns_and_risk(self):
        idx = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
        price = pd.Series(np.linspace(100, 120, 100), index=idx)
        result = MultiTimeframeCryptoAnalyzer.forward_returns(price, (4, 12))
        self.assertIn("Forward_Return_4H", result)
        metrics = MultiTimeframeCryptoAnalyzer.risk_metrics(result["Forward_Return_4H"])
        self.assertGreater(metrics["Observations"], 0)

    def test_confluence_uses_only_available_timeframes(self):
        analyzer = MultiTimeframeCryptoAnalyzer()
        idx = pd.date_range("2025-01-01", periods=140, freq="W-FRI", tz="UTC")
        base = pd.DataFrame({
            "VFI": np.linspace(-2, 2, 140),
            "VFI_Slope": np.ones(140),
            "VFI_Slope_4W": np.ones(140),
            "VFI_Slope_8W": np.ones(140),
            "VFI_Distance_Zero": np.ones(140),
            "VFI_Range_Position": np.full(140, .5),
            "VFI_Recovery_From_Min_Pct": np.full(140, 20.),
            "VFI_State": ["RECOVERY"] * 140,
            "VFI_Cycle_Stage": ["BULLISH_FLOW"] * 140,
            "VFI_Recovery_Score": np.full(140, 70.),
        }, index=idx)
        result = analyzer.confluence_at({"1W": base})
        self.assertEqual(result["Direction"], "BULLISH")
        self.assertGreaterEqual(result["Score"], 99)
