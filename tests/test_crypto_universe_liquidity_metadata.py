import unittest

from scripts.run_crypto_accumulation_breakout_screen import universe_liquidity_metadata


class FakeBinanceService:
    SPOT_BASE = "spot"
    FUTURES_BASE = "futures"

    def _get(self, base, path, params):
        if base == "spot":
            return [
                {"symbol": "AAAUSDT", "quoteVolume": "300"},
                {"symbol": "BBBUSDT", "quoteVolume": "100"},
                {"symbol": "CCCUSDT", "quoteVolume": "200"},
            ]
        return [
            {"symbol": "AAAUSDT", "quoteVolume": "50"},
            {"symbol": "BBBUSDT", "quoteVolume": "500"},
            {"symbol": "CCCUSDT", "quoteVolume": "250"},
        ]


class TestCryptoUniverseLiquidityMetadata(unittest.TestCase):
    def test_selected_assets_keep_both_market_volumes_and_ranks(self):
        result = universe_liquidity_metadata(
            FakeBinanceService(), ["AAAUSDT", "BBBUSDT", "UNKNOWNUSDT"]
        )

        self.assertEqual(result["AAAUSDT"]["Spot_Quote_Volume_24H_USDT"], 300.0)
        self.assertEqual(result["AAAUSDT"]["Spot_Quote_Volume_Rank"], 1)
        self.assertEqual(result["AAAUSDT"]["Perpetual_Quote_Volume_24H_USDT"], 50.0)
        self.assertEqual(result["AAAUSDT"]["Perpetual_Quote_Volume_Rank"], 3)
        self.assertEqual(result["BBBUSDT"]["Spot_Quote_Volume_Rank"], 3)
        self.assertEqual(result["BBBUSDT"]["Perpetual_Quote_Volume_Rank"], 1)
        self.assertEqual(result["AAAUSDT"]["Combined_Liquidity_Rank_Sum"], 4)
        self.assertTrue(result["UNKNOWNUSDT"]["Spot_Quote_Volume_Rank"] != result["UNKNOWNUSDT"]["Spot_Quote_Volume_Rank"])


if __name__ == "__main__":
    unittest.main()
