from argparse import ArgumentParser
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.binance_market_data import BinanceMarketDataService
from app.research.multi_timeframe_crypto import MultiTimeframeCryptoAnalyzer

def alignment_fields(prefix, confluence):
    fields = {f"{prefix}_Snapshot_Timestamp": confluence.get("Timestamp")}
    for row in confluence.get("Alignment", []):
        timeframe = row.get("Timeframe", "UNKNOWN")
        fields[f"{prefix}_{timeframe}_Stage"] = row.get("Stage")
        fields[f"{prefix}_{timeframe}_Timestamp"] = row.get("Timestamp")
        fields[f"{prefix}_{timeframe}_VFI"] = row.get("VFI")
        fields[f"{prefix}_{timeframe}_Slope_1B"] = row.get("Slope_1B")
        fields[f"{prefix}_{timeframe}_Slope_4B"] = row.get("Slope_4B")
        fields[f"{prefix}_{timeframe}_Slope_8B"] = row.get("Slope_8B")
        fields[f"{prefix}_{timeframe}_Recovery_Score"] = row.get("Recovery_Score")
    return fields


def main():
    parser = ArgumentParser(description="Binance Spot + USDⓈ-M perpetual multi-timeframe VFI research.")
    parser.add_argument("--symbols", default="AUTO")
    parser.add_argument("--min-volume", type=float, default=10_000_000)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--output-dir", default="reports/crypto_mtf")
    args = parser.parse_args()

    service = BinanceMarketDataService()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.symbols == "AUTO":
        spot = set(service.discover_spot_usdt(args.min_volume))
        perp = set(service.discover_usdt_perpetuals(args.min_volume))
        symbols = sorted(spot & perp, key=lambda s: s)

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    analyzer = MultiTimeframeCryptoAnalyzer()
    rows = []
    derivative_rows = []
    errors = []

    for symbol in symbols[:args.top]:
        print(f"ANALYZING {symbol}")
        try:
            spot = service.get_multi_timeframe(symbol, "spot")
            perp = service.get_multi_timeframe(symbol, "futures")
            spot_calc = analyzer.calculate(spot)
            perp_calc = analyzer.calculate(perp)
            spot_conf = analyzer.confluence_at(spot_calc)
            perp_conf = analyzer.confluence_at(perp_calc)
            row = {
                "Symbol": symbol,
                "Spot_MTF_Score": spot_conf["Score"],
                "Spot_Direction": spot_conf["Direction"],
                "Perp_MTF_Score": perp_conf["Score"],
                "Perp_Direction": perp_conf["Direction"],
                "Confluence_Score": (spot_conf["Score"] * 0.60 + perp_conf["Score"] * 0.40),
                "Spot_1W": spot_conf["Alignment"][0]["Stage"] if spot_conf["Alignment"] else None,
                "Perp_1W": perp_conf["Alignment"][0]["Stage"] if perp_conf["Alignment"] else None,
            }
            row.update(alignment_fields("Spot", spot_conf))
            row.update(alignment_fields("Perp", perp_conf))
            rows.append(row)
            oi = service.get_open_interest_history(symbol, period="1h")
            funding = service.get_funding_history(symbol)
            if not oi.empty:
                oi["OI_Change_24H"] = oi["sumOpenInterestValue"].pct_change(24)
                derivative_rows.append({
                    "Symbol": symbol,
                    "OI_Last_USD": float(oi["sumOpenInterestValue"].iloc[-1]),
                    "OI_24H_Change": float(oi["OI_Change_24H"].iloc[-1]) if pd.notna(oi["OI_Change_24H"].iloc[-1]) else None,
                    "Funding_Recent": float(funding["fundingRate"].iloc[-1]) if not funding.empty else None,
                    "Funding_Mean": float(funding["fundingRate"].tail(10).mean()) if not funding.empty else None,
                })
        except Exception as exc:
            errors.append((symbol, repr(exc)))
            print(f"ERROR {symbol}: {exc}")

    ranking = pd.DataFrame(rows)
    ranking["Universe"] = "CURRENT_BINANCE_LIQUIDITY_UNIVERSE" if not ranking.empty else pd.Series(dtype=str)
    if not ranking.empty:
        ranking["Derivatives_Adjustment"] = 0.0
        # Funding extremes reduce confidence; moderate/neutral funding does not.
        funding_map = {r["Symbol"]: r for r in derivative_rows}
        for i, row in ranking.iterrows():
            d = funding_map.get(row["Symbol"], {})
            funding = d.get("Funding_Mean")
            oi_change = d.get("OI_24H_Change")
            adjustment = 0.0
            if funding is not None and abs(funding) > 0.001:
                adjustment -= 5.0
            if oi_change is not None and abs(oi_change) > 0.10:
                adjustment -= 3.0
            ranking.loc[i, "Derivatives_Adjustment"] = adjustment
        ranking["Final_Risk_Adjusted_Score"] = (
            ranking["Confluence_Score"] + ranking["Derivatives_Adjustment"]
        ).clip(0, 100)
        ranking = ranking.sort_values("Final_Risk_Adjusted_Score", ascending=False).reset_index(drop=True)
        ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    ranking.to_csv(output / "crypto_mtf_ranking.csv", index=False)
    pd.DataFrame(derivative_rows).to_csv(output / "derivatives_snapshot.csv", index=False)
    print(ranking.to_string(index=False))
    if errors or ranking.empty:
        print(f"ANALYSIS FAILED: {len(errors)} symbol error(s); successful rows={len(ranking)}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
