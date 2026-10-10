import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_final_crypto_ranking import build_final_ranking


class FinalCryptoRankingTests(unittest.TestCase):
    def test_ranking_uses_primary_oos_strategy_and_keeps_weak_evidence_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "universe_meta").mkdir()
            pd.DataFrame([
                {"Rank": 1, "Symbol": "AAAUSDT", "Spot_MTF_Score": 80,
                 "Perp_MTF_Score": 80, "Confluence_Score": 80,
                 "Final_Risk_Adjusted_Score": 80, "Spot_Direction": "BULLISH",
                 "Perp_Direction": "BULLISH"},
                {"Rank": 1, "Symbol": "BBBUSDT", "Spot_MTF_Score": 90,
                 "Perp_MTF_Score": 90, "Confluence_Score": 90,
                 "Final_Risk_Adjusted_Score": 90, "Spot_Direction": "BULLISH",
                 "Perp_Direction": "BULLISH"},
                {"Rank": 1, "Symbol": "CCCUSDT", "Spot_MTF_Score": 99,
                 "Perp_MTF_Score": 99, "Confluence_Score": 99,
                 "Final_Risk_Adjusted_Score": 99, "Spot_Direction": "BULLISH",
                 "Perp_Direction": "BULLISH"},
            ]).to_csv(root / "crypto_mtf_ranking.csv", index=False)
            pd.DataFrame([
                {"Symbol": "AAAUSDT", "Funding_Recent": 0.0001},
                {"Symbol": "BBBUSDT", "Funding_Recent": 0.0002},
                {"Symbol": "CCCUSDT", "Funding_Recent": 0.0001},
            ]).to_csv(root / "derivatives_snapshot.csv", index=False)
            pd.DataFrame([
                {"Symbol": "AAAUSDT", "Status": "completed", "History_Bars": 3000,
                 "OOS_Folds": 3, "OOS_Trades": 140, "Reason": "", "Error": ""},
                {"Symbol": "BBBUSDT", "Status": "completed", "History_Bars": 3000,
                 "OOS_Folds": 3, "OOS_Trades": 140, "Reason": "", "Error": ""},
                {"Symbol": "CCCUSDT", "Status": "no_oos_trades", "History_Bars": 3000,
                 "OOS_Folds": 3, "OOS_Trades": 0, "Reason": "No trades", "Error": ""},
            ]).to_csv(root / "backtest_status_by_symbol.csv", index=False)

            fold_rows = []
            trade_rows = []
            for symbol, returns in [
                ("AAAUSDT", [0.01] * 20),
                ("BBBUSDT", [-0.02] * 10 + [0.005] * 10),
            ]:
                for fold in (1, 2):
                    fold_returns = returns[:10] if fold == 1 else returns[10:]
                    fold_rows.append({
                        "Symbol": symbol, "Strategy": "SPOT_MTF", "Fold": fold,
                        "Trades": len(fold_returns),
                        "Total_Return_Pct": ((1 + pd.Series(fold_returns)).prod() - 1) * 100,
                        "Max_Drawdown_Pct": -5 if symbol == "AAAUSDT" else -20,
                        "Selected_Threshold": 70,
                        "Threshold_Selection_Status": "TRAINED",
                    })
                    for i, ret in enumerate(fold_returns):
                        trade_rows.append({
                            "Symbol": symbol, "Strategy": "SPOT_MTF", "Fold": fold,
                            "Net_Return": ret,
                            "Exit_Date": f"2026-01-{fold * 10 + i + 1:02d}T00:00:00Z",
                        })
                    bh_return = -0.02 if symbol == "AAAUSDT" else 0.02
                    fold_rows.append({
                        "Symbol": symbol, "Strategy": "BUY_AND_HOLD_OOS", "Fold": fold,
                        "Trades": 1, "Total_Return_Pct": bh_return * 100,
                        "Max_Drawdown_Pct": -10, "Selected_Threshold": 0,
                        "Threshold_Selection_Status": "BENCHMARK_NO_TUNING",
                    })
                    trade_rows.append({
                        "Symbol": symbol, "Strategy": "BUY_AND_HOLD_OOS", "Fold": fold,
                        "Net_Return": bh_return,
                        "Exit_Date": f"2026-01-{fold * 10 + 1:02d}T00:00:00Z",
                    })
            for fold in (1, 2):
                fold_rows.append({
                    "Symbol": "CCCUSDT", "Strategy": "SPOT_MTF", "Fold": fold,
                    "Trades": 0, "Total_Return_Pct": 0, "Max_Drawdown_Pct": 0,
                    "Selected_Threshold": 80,
                    "Threshold_Selection_Status": "FALLBACK_NO_TRAIN_TRADES",
                })
            pd.DataFrame(fold_rows).to_csv(root / "walk_forward_oos.csv", index=False)
            pd.DataFrame(trade_rows).to_csv(root / "oos_trades.csv", index=False)
            (root / "universe_meta" / "reports-universe.json").write_text(json.dumps({
                "retrieved_at_utc": "2026-10-10T00:00:00Z",
                "selection_method": "test liquidity rank",
                "symbols": ["AAAUSDT", "BBBUSDT", "CCCUSDT"],
                "records": [
                    {"Symbol": "AAAUSDT", "Spot_Volume_Rank": 1, "Perp_Volume_Rank": 2,
                     "Spot_24h_Quote_Volume_USDT": 1000000, "Perp_24h_Quote_Volume_USDT": 900000},
                    {"Symbol": "BBBUSDT", "Spot_Volume_Rank": 2, "Perp_Volume_Rank": 1,
                     "Spot_24h_Quote_Volume_USDT": 900000, "Perp_24h_Quote_Volume_USDT": 1000000},
                    {"Symbol": "CCCUSDT", "Spot_Volume_Rank": 3, "Perp_Volume_Rank": 3,
                     "Spot_24h_Quote_Volume_USDT": 500000, "Perp_24h_Quote_Volume_USDT": 450000},
                ],
            }), encoding="utf-8")

            build_final_ranking(root)

            result = pd.read_csv(root / "final_crypto_ranking.csv")
            self.assertEqual(result.iloc[0]["Symbol"], "AAAUSDT")
            self.assertEqual(result.iloc[-1]["Symbol"], "CCCUSDT")
            aaa = result.set_index("Symbol").loc["AAAUSDT"]
            self.assertEqual(int(aaa["OOS_Trades"]), 20)
            self.assertEqual(int(aaa["All_Strategy_OOS_Trades"]), 140)
            self.assertEqual(int(aaa["Spot_Volume_Rank"]), 1)
            self.assertIn("matched spot buy-and-hold", aaa["Research_Reason"])
            self.assertIn("Liquidez", aaa["Motivo_Ranking_ES"])
            self.assertEqual(aaa["Evidence_Tier"], "POSITIVE_OOS_EVIDENCE")

            by_symbol = pd.read_csv(root / "walk_forward_summary_by_symbol.csv")
            self.assertIn("Symbol", by_symbol.columns)
            self.assertIn("BUY_AND_HOLD_OOS", set(by_symbol["Strategy"]))
            audit = json.loads((root / "data_quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["unique_symbols_in_final_ranking"], 3)


if __name__ == "__main__":
    unittest.main()
