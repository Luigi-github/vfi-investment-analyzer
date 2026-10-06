from argparse import ArgumentParser
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.binance_market_data import BinanceMarketDataService
from app.research.multi_timeframe_crypto import MultiTimeframeCryptoAnalyzer

def summarize(signals, return_col):
    x = signals.dropna(subset=[return_col]).copy()
    if x.empty:
        return {"Observations": 0}
    r = x[return_col]
    return {
        "Observations": int(len(r)),
        "Positive_Rate_Pct": float((r > 0).mean() * 100),
        "Average_Return_Pct": float(r.mean() * 100),
        "Median_Return_Pct": float(r.median() * 100),
        "Worst_Return_Pct": float(r.min() * 100),
        "Best_Return_Pct": float(r.max() * 100),
    }

def main():
    p = ArgumentParser(description="Historical 1H confluence backtest: Spot + perpetual VFI.")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT")
    p.add_argument("--output-dir", default="reports/crypto_mtf_backtest")
    args = p.parse_args()

    service = BinanceMarketDataService()
    analyzer = MultiTimeframeCryptoAnalyzer()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary_rows = []

    for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        print(f"BACKTEST {symbol}")
        try:
            spot = analyzer.calculate(service.get_multi_timeframe(symbol, "spot"))
            perp = analyzer.calculate(service.get_multi_timeframe(symbol, "futures"))
            spot_hist = analyzer.build_1h_confluence_history(spot)
            perp_hist = analyzer.build_1h_confluence_history(perp)
            if spot_hist.empty or perp_hist.empty:
                continue
            merged = pd.merge_asof(
                spot_hist.reset_index().sort_values("Timestamp"),
                perp_hist.reset_index().sort_values("Timestamp"),
                on="Timestamp", suffixes=("_Spot", "_Perp"), direction="backward",
            ).set_index("Timestamp")
            merged["Symbol"] = symbol
            merged["Combined_MTF_Score"] = merged["MTF_Score_Spot"] * .60 + merged["MTF_Score_Perp"] * .40
            merged["Both_Bullish"] = (merged["MTF_Score_Spot"] >= 70) & (merged["MTF_Score_Perp"] >= 70)
            merged["Spot_Bullish"] = merged["MTF_Score_Spot"] >= 70
            merged["Perp_Bullish"] = merged["MTF_Score_Perp"] >= 70
            for bars in (4, 12, 24, 72):
                merged[f"Forward_Return_{bars}H"] = merged["Close_Spot"].shift(-bars) / merged["Close_Spot"] - 1
            merged.to_csv(output / f"{symbol}_history.csv")
            for name, mask in {
                "SPOT_MTF": merged["Spot_Bullish"],
                "PERP_MTF": merged["Perp_Bullish"],
                "BOTH_MTF": merged["Both_Bullish"],
                "COMBINED_70": merged["Combined_MTF_Score"] >= 70,
                "COMBINED_80": merged["Combined_MTF_Score"] >= 80,
            }.items():
                for horizon in (4, 12, 24, 72):
                    stats = summarize(merged.loc[mask], f"Forward_Return_{horizon}H")
                    stats.update({"Symbol": symbol, "Strategy": name, "Horizon_H": horizon})
                    summary_rows.append(stats)
            all_rows.append(merged)
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "strategy_comparison.csv", index=False)
    if not summary.empty:
        summary.sort_values(["Horizon_H", "Average_Return_Pct"], ascending=[True, False]).to_csv(
            output / "strategy_comparison_sorted.csv", index=False
        )
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
