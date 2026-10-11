"""Screen 100 highly liquid US-listed stocks and ETFs on completed daily/weekly bars.

Combines the project's daily VFI zero-approach setup, weekly MM volume divergence,
and OBV/A-D/CMF/MFI pressure indicators. This is a curated liquid universe, not a
claim to dynamically rank every US listing by today's traded dollar volume.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.data.market_data import MarketDataService
from run_example_assets_accumulation_screen import (
    analyze_symbol as analyze_vfi_divergence,
    completed_daily,
    completed_weekly,
)
from run_volume_accumulation_indicators import analyze_symbol as analyze_volume_indicators

US_LIQUID_100 = """
AAPL MSFT NVDA AMZN GOOGL GOOG META TSLA AVGO AMD INTC MU QCOM TXN ORCL CRM
PLTR NFLX ADBE CSCO IBM NOW PANW CRWD SMCI ARM AMAT LRCX KLAC TSM JPM BAC WFC
GS MS C SCHW BLK V MA AXP COIN HOOD PYPL XYZ SOFI NU LLY NVO UNH JNJ ABBV MRK
PFE AMGN GILD ISRG WMT COST HD LOW MCD SBUX NKE KO PG XOM CVX COP SLB OXY CAT
GE BA HON UPS F GM DE SPY QQQ IWM DIA VTI XLF XLK XLE TLT GLD SLV HYG USO XBI
ARKK SMH SOXX IBIT EEM
""".split()

VFI_EARLY_TYPES = {"PRE_ZERO_BREAK", "ZERO_CROSS_RECENT"}


def as_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def early_score(row: dict) -> float:
    # Missing components are excluded and weights renormalized.
    components = [
        (as_float(row.get("VFI_1D_Setup_Score")), 0.25),
        (as_float(row.get("Divergence_1W_Setup_Score")), 0.20),
        (as_float(row.get("Accumulation_Distribution_Score")), 0.25),
        (as_float(row.get("Weekly_Accumulation_Distribution_Score")), 0.20),
        (as_float(row.get("Price_Consolidation_Score")), 0.10),
    ]
    valid = [(score, weight) for score, weight in components if np.isfinite(score)]
    if not valid:
        return 0.0
    return float(sum(score * weight for score, weight in valid) / sum(weight for _, weight in valid))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rank 100 liquid US-listed assets using daily VFI and weekly volume-flow confirmation."
    )
    parser.add_argument("--period", default="max")
    parser.add_argument("--output-dir", default="reports/us_liquid_100")
    parser.add_argument("--symbols", default=",".join(US_LIQUID_100))
    args = parser.parse_args()

    symbols = list(dict.fromkeys(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()))
    if not symbols:
        parser.error("Provide at least one symbol")

    service = MarketDataService()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, symbol in enumerate(symbols, start=1):
        print(f"[{index}/{len(symbols)}] {symbol}", flush=True)
        try:
            raw = service.get_daily_data(symbol, period=args.period)
            daily = completed_daily(raw)
            weekly = completed_weekly(daily)
            vfi_row = analyze_vfi_divergence(symbol, daily, weekly)
            flow_row = analyze_volume_indicators(symbol, daily)
            row = dict(vfi_row)

            # Keep the daily VFI/weekly divergence results and add both timeframe
            # versions of OBV, A/D, CMF and MFI without overwriting VFI status.
            flow_keys = [
                "Accumulation_Distribution_Score", "Distribution_Pressure_Score",
                "Pressure_Classification", "Potential_Stealth_Accumulation",
                "OBV_Trend_Score", "AD_Trend_Score", "CMF_20", "MFI_14",
                "Relative_Volume_20D", "Price_Return_20B", "Price_Return_60B",
                "Weekly_Accumulation_Distribution_Score",
                "Weekly_Distribution_Pressure_Score", "Weekly_Pressure_Classification",
                "Weekly_OBV_Trend_Score", "Weekly_AD_Trend_Score",
                "Weekly_CMF_20", "Weekly_MFI_14",
            ]
            for key in flow_keys:
                if key in flow_row:
                    row[key] = flow_row[key]

            if len(daily) >= 20:
                dollar_volume = (daily["Close"] * daily["Volume"]).tail(20).mean()
            else:
                dollar_volume = np.nan
            row["Avg_Dollar_Volume_20D_USD"] = as_float(dollar_volume)
            return20 = as_float(row.get("Price_Return_20B"))
            # Prefer flat or moderately declining prices over already extended rallies.
            row["Price_Consolidation_Score"] = (
                float(np.clip(100 - max(return20 * 100, 0) * 6 - abs(min(return20 * 100, 0)) * 3, 0, 100))
                if np.isfinite(return20) else np.nan
            )
            vfi_type = str(row.get("VFI_Setup_Type", ""))
            div_type = str(row.get("Divergence_Setup_Type", ""))
            daily_flow = as_float(row.get("Accumulation_Distribution_Score"))
            weekly_flow = as_float(row.get("Weekly_Accumulation_Distribution_Score"))
            row["Early_Accumulation_Score"] = early_score(row)
            row["Early_Accumulation_Candidate"] = bool(
                (vfi_type in VFI_EARLY_TYPES or div_type in VFI_EARLY_TYPES)
                and np.isfinite(daily_flow) and daily_flow >= 50
                and np.isfinite(weekly_flow) and weekly_flow >= 50
                and np.isfinite(return20) and return20 <= 0.05
            )
            row["Daily_Flow_Status"] = "AVAILABLE" if np.isfinite(daily_flow) else "UNAVAILABLE"
            row["Weekly_Flow_Status"] = "AVAILABLE" if np.isfinite(weekly_flow) else "UNAVAILABLE"
            metadata = service.get_last_metadata()
            row["Data_Provider"] = metadata.get("Provider", "Yahoo Finance / yfinance")
            row["Data_First_Date"] = metadata.get("First_Date")
            row["Data_Last_Date"] = metadata.get("Last_Date")
            row["Data_SHA256"] = metadata.get("Data_SHA256")
        except Exception as exc:
            row = {
                "Symbol": symbol,
                "Analysis_Status": "ERROR",
                "Error": repr(exc),
                "Early_Accumulation_Score": 0.0,
                "Early_Accumulation_Candidate": False,
            }
        row["Universe_Order"] = index
        rows.append(row)

    result = pd.DataFrame(rows)
    if "Early_Accumulation_Score" not in result:
        result["Early_Accumulation_Score"] = 0.0
    result = result.sort_values(
        ["Early_Accumulation_Candidate", "Early_Accumulation_Score", "Avg_Dollar_Volume_20D_USD"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))

    result.to_csv(output_dir / "us_liquid_100_daily_weekly_ranking.csv", index=False)
    result[result.get("Early_Accumulation_Candidate", pd.Series(False, index=result.index)).fillna(False)].to_csv(
        output_dir / "early_accumulation_candidates.csv", index=False
    )
    result[result.get("Analysis_Status", pd.Series("", index=result.index)) == "ERROR"].to_csv(
        output_dir / "analysis_errors.csv", index=False
    )
    summary = {
        "universe": "Curated US-listed liquid stocks and ETFs; not a live all-market liquidity census",
        "symbols_requested": symbols,
        "assets_reported": int(len(result)),
        "analysis_status_counts": result.get("Analysis_Status", pd.Series(dtype=str)).value_counts().to_dict(),
        "daily_timeframe": "completed daily candles; VFI + OBV/A-D/CMF/MFI",
        "weekly_timeframe": "completed W-FRI candles; Volume Divergence by MM + OBV/A-D/CMF/MFI",
        "early_accumulation_candidates": int(result.get("Early_Accumulation_Candidate", pd.Series(dtype=bool)).fillna(False).sum()),
        "ranking": "Early-accumulation score combines daily VFI setup, weekly divergence setup, daily/weekly flow scores and price consolidation; candidates require a VFI/divergence early setup, daily and weekly flow >=50, and 20-bar price return <=5%.",
        "data_provider": "Yahoo Finance via yfinance",
        "warning": "Volume-based indicators are price/volume proxies and do not identify institutional orders or guarantee future price increases.",
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    columns = [
        "Rank", "Symbol", "Analysis_Status", "Early_Accumulation_Candidate",
        "Early_Accumulation_Score", "Price", "Price_Return_20B",
        "VFI_1D_Value", "VFI_1D_Slope_4B", "VFI_1D_Slope_8B", "VFI_Setup_Type",
        "Divergence_1W_Value", "Divergence_1W_Slope_4B", "Divergence_1W_Slope_8B",
        "Divergence_Setup_Type", "Accumulation_Distribution_Score",
        "Weekly_Accumulation_Distribution_Score", "CMF_20", "MFI_14",
        "Avg_Dollar_Volume_20D_USD", "Daily_Candle_Date", "Weekly_Candle_Date",
    ]
    print(json.dumps(summary, indent=2, default=str))
    print(result[[column for column in columns if column in result.columns]].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
