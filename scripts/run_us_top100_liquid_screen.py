"""Screen the 100 most liquid names from a broad US large-cap universe.

Liquidity is ranked by mean daily dollar turnover (Close * Volume) over the
latest 20 completed sessions. The curated seed universe is, so this
is a reproducible liquid-US-universe screen, not a claim to cover every US
listing. Ranking emphasizes pre-zero VFI, weekly MM volume divergence and
OHLCV accumulation proxies (OBV, A/D, CMF, MFI). No intraday bars are used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.market_data import MarketDataService
from app.research.accumulation_breakout_screen import rank_accumulation_candidates
from scripts.run_example_assets_accumulation_screen import (
    analyze_symbol as analyze_vfi_divergence,
    completed_daily,
    completed_weekly,
)
from scripts.run_volume_accumulation_indicators import calculate_indicators

# Broad candidate pool of liquid US-listed stocks. Actual 20-session dollar
# turnover decides which 100 are analyzed, rather than this list's order.
US_CANDIDATE_UNIVERSE = [
    "AAPL","NVDA","MSFT","AMZN","GOOGL","GOOG","META","AVGO","TSLA","AMD",
    "BRK-B","LLY","JPM","WMT","XOM","V","MA","COST","NFLX","ORCL",
    "PLTR","BAC","ABBV","HD","PG","JNJ","CVX","KO","GE","CSCO",
    "CRM","CAT","MRK","WFC","IBM","MU","PM","GS","AXP","UNH",
    "RTX","TMO","MS","LIN","MCD","PEP","ABT","DIS","UBER","NOW",
    "INTC","QCOM","TXN","AMAT","LRCX","KLAC","ANET","PANW","CRWD","ADBE",
    "ISRG","AMGN","GILD","PFE","BSX","VRTX","REGN","SYK","MDT","CI",
    "C","SCHW","BLK","CME","ICE","COIN","HOOD","SOFI","AXP","USB",
    "CVS","LOW","TGT","TJX","NKE","SBUX","CMG","YUM","BKNG","ABNB",
    "F","GM","DE","HON","ETN","PH","UNP","UPS","FDX","BA",
    "COP","SLB","EOG","OXY","PSX","VLO","MPC","CEG","NEE","DUK",
    "SO","PLD","AMT","WELL","GSAT","T","VZ","TMUS","CHTR","CMCSA",
    "SHOP","SNOW","DDOG","NET","MSTR","MELI","PDD","BABA","TSM","ASML",
    "BIDU","NEM","FCX","CLF","RIVN","LCID","CVNA","DASH","APP","ARM",
]

def dollar_turnover(data: pd.DataFrame) -> float:
    data = completed_daily(data)
    if data.empty or not {"Close", "Volume"}.issubset(data.columns):
        return float("nan")
    values = pd.to_numeric(data["Close"], errors="coerce") * pd.to_numeric(data["Volume"], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna().tail(20)
    return float(values.mean()) if len(values) >= 10 else float("nan")


def early_price_score(ret20: float) -> float:
    """Prefer a flat-to-down price over an already extended move."""
    if not np.isfinite(ret20):
        return 50.0
    if ret20 <= 0:
        return float(np.clip(70.0 + min(abs(ret20), 0.20) * 100.0, 70, 90))
    if ret20 <= 0.05:
        return 70.0 - ret20 * 200.0
    if ret20 <= 0.15:
        return 60.0 - (ret20 - 0.05) * 300.0
    return float(max(0.0, 30.0 - (ret20 - 0.15) * 200.0))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="max")
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--output-dir", default="reports/us_top100_liquid")
    args = parser.parse_args()
    symbols = list(dict.fromkeys(US_CANDIDATE_UNIVERSE))
    service = MarketDataService()
    raw_data: dict[str, pd.DataFrame] = {}
    liquidity_rows = []
    print(f"Downloading completed daily candles for {len(symbols)} US-listed candidates...", flush=True)
    for i, symbol in enumerate(symbols, 1):
        print(f"[universe {i}/{len(symbols)}] {symbol}", flush=True)
        try:
            raw = service.get_daily_data(symbol, period=args.period)
            daily = completed_daily(raw)
            raw_data[symbol] = daily
            turnover = dollar_turnover(daily)
            liquidity_rows.append({
                "Symbol": symbol,
                "Average_Dollar_Turnover_20D": turnover,
                "Daily_Bars": len(daily),
                "Data_Last_Date": daily.index[-1].strftime("%Y-%m-%d") if not daily.empty else "",
                "Data_Status": "OK" if np.isfinite(turnover) else "INSUFFICIENT_DATA",
            })
        except Exception as exc:
            liquidity_rows.append({
                "Symbol": symbol, "Average_Dollar_Turnover_20D": np.nan,
                "Daily_Bars": 0, "Data_Last_Date": "", "Data_Status": f"ERROR: {exc!r}",
            })
    liquidity = pd.DataFrame(liquidity_rows)
    eligible = liquidity[np.isfinite(pd.to_numeric(liquidity["Average_Dollar_Turnover_20D"], errors="coerce"))]
    selected = eligible.sort_values("Average_Dollar_Turnover_20D", ascending=False).head(args.top)
    selected_symbols = selected["Symbol"].tolist()
    print(f"Selected {len(selected_symbols)} names by 20-session average dollar turnover.", flush=True)

    rows = []
    for i, symbol in enumerate(selected_symbols, 1):
        print(f"[analysis {i}/{len(selected_symbols)}] {symbol}", flush=True)
        daily = raw_data[symbol]
        weekly = completed_weekly(daily)
        row = analyze_vfi_divergence(symbol, daily, weekly)
        row["Average_Dollar_Turnover_20D"] = float(
            selected.loc[selected["Symbol"] == symbol, "Average_Dollar_Turnover_20D"].iloc[0]
        )
        row["Liquidity_Rank"] = i
        try:
            vol = calculate_indicators(daily)
            row.update({f"Flow_{k}": v for k, v in vol.items()})
        except Exception as exc:
            row["Flow_Analysis_Status"] = f"ERROR: {exc!r}"
        row["Early_Price_Score"] = early_price_score(float(row.get("Flow_Price_Return_20B", np.nan)))
        rows.append(row)

    result = rank_accumulation_candidates(pd.DataFrame(rows))
    # Blend the existing rule-based setup score with independent volume-flow
    # evidence and a penalty for already extended prices. Keep Setup_Tier first.
    flow_score = pd.to_numeric(
        result.get("Flow_Accumulation_Distribution_Score", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    vfi_score = pd.to_numeric(result.get("VFI_Setup_Score", 0), errors="coerce").fillna(0)
    div_score = pd.to_numeric(result.get("Divergence_Setup_Score", 0), errors="coerce").fillna(0)
    price_score = pd.to_numeric(result.get("Early_Price_Score", 50), errors="coerce").fillna(50)
    # Reweight when flow data are missing; missing flow must not count as zero.
    result["Early_Accumulation_Score"] = (
        0.45 * vfi_score + 0.25 * div_score + 0.15 * price_score
        + 0.15 * flow_score.fillna(50)
    ).clip(0, 100)
    result["Price_Not_Extended"] = pd.to_numeric(
        result.get("Flow_Price_Return_20B", pd.Series(np.nan, index=result.index)), errors="coerce"
    ).le(0.05)
    result = result.sort_values(
        ["Setup_Tier", "Early_Accumulation_Score", "Average_Dollar_Turnover_20D"],
        ascending=[True, False, False], na_position="last", kind="mergesort"
    ).reset_index(drop=True)
    result.insert(0, "Opportunity_Rank", range(1, len(result) + 1))
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "us_top100_early_accumulation_ranking.csv", index=False)
    liquidity.sort_values("Average_Dollar_Turnover_20D", ascending=False).to_csv(
        out / "us_liquidity_universe.csv", index=False
    )
    result[result.get("Combined_Setup", pd.Series(False, index=result.index)).fillna(False)].to_csv(
        out / "us_top100_combined_candidates.csv", index=False
    )
    summary = {
        "candidate_universe_size": len(symbols),
        "requested_top": args.top,
        "selected_and_analyzed": len(result),
        "analysis_status_counts": result.get("Analysis_Status", pd.Series(dtype=str)).value_counts().to_dict(),
        "completed_daily_candles_only": True,
        "weekly_divergence_timeframe": "W-FRI completed candles",
        "liquidity_method": "Mean Close * Volume over latest 20 completed daily candles, ranked descending within the curated US-listed candidate universe",
        "early_accumulation_ranking": "Existing VFI/divergence setup tier first, then 45% VFI score + 25% weekly divergence score + 15% flat/down price score + 15% OBV/A-D/CMF/MFI composite",
        "caveat": "This ranks technical proxies, not confirmed institutional transactions; universe is curated, not every US listing.",
        "data_cutoff": str(result.get("Daily_Candle_Date", pd.Series(dtype=str)).max()),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    display_cols = [
        "Opportunity_Rank", "Symbol", "Price", "Average_Dollar_Turnover_20D",
        "Setup_Tier", "VFI_Setup_Type", "VFI_Setup_Score",
        "Divergence_Setup_Type", "Divergence_Setup_Score",
        "Flow_Accumulation_Distribution_Score", "Flow_Price_Return_20B",
        "Early_Accumulation_Score", "Combined_Setup", "Analysis_Status",
    ]
    print(json.dumps(summary, indent=2, default=str))
    print(result[[c for c in display_cols if c in result.columns]].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
