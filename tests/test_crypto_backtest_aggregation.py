import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_resumable_crypto_backtests import aggregate_outputs


class TestCryptoBacktestAggregation(unittest.TestCase):
    def test_global_mtf_ranking_recalculates_rank_across_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = [
                ("LOWUSDT", 1, 40.0, 45.0),
                ("HIGHUSDT", 1, 90.0, 92.0),
                ("MIDUSDT", 1, 70.0, 72.0),
            ]
            for symbol, rank, final_score, confluence in samples:
                folder = root / "per_symbol" / symbol / "mtf"
                folder.mkdir(parents=True)
                pd.DataFrame([{
                    "Rank": rank,
                    "Symbol": symbol,
                    "Confluence_Score": confluence,
                    "Final_Risk_Adjusted_Score": final_score,
                }]).to_csv(folder / "crypto_mtf_ranking.csv", index=False)

            aggregate_outputs(root)
            ranking = pd.read_csv(root / "crypto_mtf_ranking.csv")

            self.assertEqual(ranking["Symbol"].tolist(), ["HIGHUSDT", "MIDUSDT", "LOWUSDT"])
            self.assertEqual(ranking["Rank"].tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
