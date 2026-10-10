import unittest

from scripts.run_crypto_mtf_analysis import alignment_fields, alignment_stage


class MtfTimeframeLabelTests(unittest.TestCase):
    def test_missing_weekly_data_is_not_mislabeled_as_weekly(self):
        confluence = {
            "Timestamp": "2026-10-10T14:00:00Z",
            "Alignment": [
                {"Timeframe": "1D", "Stage": "SUSTAINED_WEAKENING",
                 "Timestamp": "2026-10-09T00:00:00Z", "VFI": 4.0,
                 "Slope_1B": -0.5, "Slope_4B": -0.2, "Slope_8B": 0.1,
                 "Recovery_Score": 20.0},
                {"Timeframe": "4H", "Stage": "BULLISH_FLOW",
                 "Timestamp": "2026-10-10T08:00:00Z", "VFI": 2.0,
                 "Slope_1B": 0.2, "Slope_4B": 0.3, "Slope_8B": 0.1,
                 "Recovery_Score": 70.0},
            ],
        }

        self.assertIsNone(alignment_stage(confluence, "1W"))
        self.assertEqual(alignment_stage(confluence, "1D"), "SUSTAINED_WEAKENING")
        fields = alignment_fields("Spot", confluence)
        self.assertIsNone(fields["Spot_1W_Stage"])
        self.assertEqual(fields["Spot_1D_Stage"], "SUSTAINED_WEAKENING")
        self.assertEqual(fields["Spot_4H_Stage"], "BULLISH_FLOW")
        self.assertIsNone(fields["Spot_1H_Stage"])


if __name__ == "__main__":
    unittest.main()
