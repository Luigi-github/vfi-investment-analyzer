"""Run the 1D VFI + 1W Volume Divergence screen on the project's example assets.

Uses Yahoo Finance data so the examples can be validated before the Binance universe scan.
Only completed daily and weekly candles are included.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.vfi_analyzer import VFIAnalyzer
from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM
from app.research.accumulation_breakout_screen import (
    rank_accumulation_candidates,
    score_zero_approach,
    unavailable_asset_row,
)
from app.scoring.vfi_score import VFIScore

# US-listed assets and crypto examples, plus Colombian equities from BVC
# Yahoo Finance uses the .CL suffix for BVC-listed shares.
DEFAULT_SYMBOLS = [
    "DDD", "INTC", "NVDA", "BTC-USD", "ETH-USD", "LTC-USD",
    "EEM", "SPY", "AMD", "SQM", "EC", "AU", "CIB", "NU",
    "ISA.CL", "GEB.CL", "MINEROS.CL",
]


def completed_daily(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    today = pd.Timestamp.now().normalize()
    return data.loc[data.index.normalize() < today].copy()


def completed_weekly(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    weekly = data.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    today = pd.Timestamp.now().normalize()
    last_friday = today - pd.Timedelta(days=(today.weekday() - 4) % 7)
    if today.weekday() == 4:
        last_friday -= pd.Timedelta(days=7)
    return weekly.loc[weekly.index.normalize() <= last_friday].copy()


def analyze_symbol(symbol: str, daily: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    row = {
        "Symbol": symbol,
        "Analysis_Status": "ERROR",
        "Exclusion_Reason": "",
        "Daily_Bars": int(len(daily)),
        "Weekly_Bars": int(len(weekly)),
    }
    errors = []

    try:
        if daily.empty:
            raise ValueError("No completed daily data")
        vfi = VFIIndicator().calculate(daily)
        vfi = VFIScore().calculate(VFIAnalyzer().analyze(vfi))
        vfi_valid = vfi.loc[vfi["VFI"].notna()]
        if vfi_valid.empty:
            raise ValueError(f"No valid VFI after warm-up ({len(daily)} daily bars)")
        latest = vfi_valid.iloc[-1]
        setup = score_zero_approach(
            vfi_valid["VFI"], "VFI_1D", lookback=365, min_history=365,
            near_zero_fraction=0.15, max_abs_near_zero=2.0,
            recent_cross_fraction=0.10, recent_cross_bars=3,
            min_strength_4=0.10, min_strength_8=0.25,
        )
        row.update({
            "Daily_Candle_Date": vfi_valid.index[-1].strftime("%Y-%m-%d"),
            "Price": float(latest["Close"]),
            "VFI_State_1D": str(latest.get("VFI_State", "UNKNOWN")),
            "VFI_Cycle_Stage_1D": str(latest.get("VFI_Cycle_Stage", "UNKNOWN")),
            "VFI_Recovery_Score_1D": float(latest.get("VFI_Recovery_Score", float("nan"))),
            "VFI_Analysis_Status": "ANALYZED",
        })
        row.update(setup)
        row["VFI_Setup_Type"] = setup["VFI_1D_Setup_Type"]
        row["VFI_Setup_Score"] = setup["VFI_1D_Setup_Score"]
    except Exception as exc:
        errors.append(f"VFI: {exc!r}")
        row["VFI_Analysis_Status"] = "ERROR"
        row.update(score_zero_approach(pd.Series(dtype=float), "VFI_1D", lookback=365, min_history=365))
        row["VFI_Setup_Type"] = "ERROR"
        row["VFI_Setup_Score"] = 0.0

    try:
        if weekly.empty:
            raise ValueError("No completed weekly data")
        divergence = VolumeDivergenceMM().calculate(weekly)
        valid = divergence.loc[divergence["Volume_Divergence_MM"].notna()]
        if valid.empty:
            raise ValueError(f"No valid weekly divergence after warm-up ({len(weekly)} weekly bars)")
        latest_div = valid.iloc[-1]
        setup = score_zero_approach(
            valid["Volume_Divergence_MM"], "Divergence_1W",
            lookback=52, min_history=20, near_zero_fraction=0.15,
            max_abs_near_zero=None, recent_cross_fraction=0.10,
            recent_cross_bars=2, min_strength_4=0.10, min_strength_8=0.25,
        )
        row.update({
            "Weekly_Candle_Date": valid.index[-1].strftime("%Y-%m-%d"),
            "Divergence_MM_Signal_1W": str(latest_div.get("Volume_Div_Signal", "")),
            "Divergence_Analysis_Status": "ANALYZED",
        })
        row.update(setup)
        row["Divergence_Setup_Type"] = setup["Divergence_1W_Setup_Type"]
        row["Divergence_Setup_Score"] = setup["Divergence_1W_Setup_Score"]
    except Exception as exc:
        errors.append(f"Divergence: {exc!r}")
        row["Divergence_Analysis_Status"] = "ERROR"
        row.update(score_zero_approach(pd.Series(dtype=float), "Divergence_1W", lookback=52, min_history=20))
        row["Divergence_Setup_Type"] = "ERROR"
        row["Divergence_Setup_Score"] = 0.0

    if row["VFI_Analysis_Status"] == "ANALYZED" and row["Divergence_Analysis_Status"] == "ANALYZED":
        row["Analysis_Status"] = "ANALYZED"
    elif row["VFI_Analysis_Status"] == "ANALYZED" or row["Divergence_Analysis_Status"] == "ANALYZED":
        row["Analysis_Status"] = "PARTIAL_ERROR"
    else:
        row["Analysis_Status"] = "ERROR"
    row["Exclusion_Reason"] = "; ".join(errors)
    return row


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Example-asset validation for daily VFI + weekly MM volume divergence.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--period", default="max")
    parser.add_argument("--output-dir", default="reports/example_assets_accumulation")
    args = parser.parse_args()

    symbols = list(dict.fromkeys(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    if not symbols:
        parser.error("Provide at least one symbol")

    service = MarketDataService()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, symbol in enumerate(symbols, start=1):
        print(f"[{i}/{len(symbols)}] {symbol}", flush=True)
        try:
            raw = service.get_daily_data(symbol, period=args.period)
            daily = completed_daily(raw)
            weekly = completed_weekly(daily)
            row = analyze_symbol(symbol, daily, weekly)
            metadata = service.get_last_metadata()
            row["Data_Provider"] = metadata.get("Provider", "Yahoo Finance / yfinance")
            row["Data_First_Date"] = metadata.get("First_Date")
            row["Data_Last_Date"] = metadata.get("Last_Date")
            row["Data_SHA256"] = metadata.get("Data_SHA256")
        except Exception as exc:
            row = unavailable_asset_row(symbol, i, "ERROR", repr(exc))
            row["Data_Provider"] = "Yahoo Finance / yfinance"
        row["Input_Order"] = i
        rows.append(row)

    result = rank_accumulation_candidates(pd.DataFrame(rows))
    if not result.empty:
        result.insert(0, "Rank", range(1, len(result) + 1))
    result.to_csv(out / "all_example_assets.csv", index=False)
    result[result.get("VFI_Setup", pd.Series(False, index=result.index)).fillna(False)].to_csv(
        out / "vfi_daily_candidates.csv", index=False
    )
    result[result.get("Combined_Setup", pd.Series(False, index=result.index)).fillna(False)].to_csv(
        out / "combined_candidates.csv", index=False
    )
    summary = {
        "symbols_requested": symbols,
        "assets_reported": int(len(result)),
        "analysis_status_counts": result["Analysis_Status"].value_counts().to_dict() if "Analysis_Status" in result else {},
        "vfi_daily_candidates": int(result["VFI_Setup"].fillna(False).sum()) if "VFI_Setup" in result else 0,
        "weekly_divergence_candidates": int(result["Divergence_Setup"].fillna(False).sum()) if "Divergence_Setup" in result else 0,
        "combined_candidates": int(result["Combined_Setup"].fillna(False).sum()) if "Combined_Setup" in result else 0,
        "timeframes": {"VFI": "1D", "Volume_Divergence_by_MM": "1W"},
        "daily_and_weekly_candles": "completed candles only",
        "ranking_rules": "Same early zero-approach rules as Binance screen; no intraday signals.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    columns = [
        "Rank", "Symbol", "Analysis_Status", "Price", "Daily_Candle_Date", "Weekly_Candle_Date",
        "VFI_1D_Value", "VFI_1D_Slope_4B", "VFI_1D_Slope_8B", "VFI_1D_Setup_Type",
        "Divergence_1W_Value", "Divergence_1W_Slope_4B", "Divergence_1W_Slope_8B",
        "Divergence_1W_Setup_Type", "Combined_Setup", "Exclusion_Reason",
    ]
    print(result[[c for c in columns if c in result.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
