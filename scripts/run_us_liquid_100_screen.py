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
from app.analysis.vfi_analyzer import VFIAnalyzer
from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM
from app.research.accumulation_breakout_screen import score_zero_approach
from app.scoring.vfi_score import VFIScore
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
PFE AMGN GILD ISRG WMT COST HD LOW MCD SBUX NKE KO PEP PG XOM CVX COP SLB OXY CAT
GE BA HON UPS F GM DE SPY QQQ IWM DIA VTI XLF XLK XLE TLT GLD SLV HYG LQD USO XBI
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
        (as_float(row.get("VFI_1W_Setup_Score")), 0.30),
        (as_float(row.get("VFI_1D_Setup_Score")), 0.20),
        (as_float(row.get("Divergence_1W_Setup_Score")), 0.20),
        (as_float(row.get("Accumulation_Distribution_Score")), 0.15),
        (as_float(row.get("Weekly_Accumulation_Distribution_Score")), 0.10),
        (as_float(row.get("Price_Consolidation_Score")), 0.05),
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
            # STAGE 1: rank all assets on completed weekly candles first.
            row = {"Symbol": symbol, "Analysis_Status": "WEEKLY_SCREENED",
                   "Weekly_Bars": int(len(weekly))}
            weekly_vfi_setup = {}
            weekly_div_setup = {}
            try:
                weekly_vfi = VFIIndicator().calculate(weekly)
                weekly_vfi = VFIScore().calculate(VFIAnalyzer().analyze(weekly_vfi))
                valid_vfi = weekly_vfi.loc[weekly_vfi["VFI"].notna()]
                if valid_vfi.empty:
                    raise ValueError("No valid weekly VFI")
                weekly_vfi_setup = score_zero_approach(
                    valid_vfi["VFI"], "VFI_1W", lookback=260, min_history=100,
                    near_zero_fraction=0.15, max_abs_near_zero=2.0,
                    recent_cross_fraction=0.10, recent_cross_bars=3,
                    min_strength_4=0.10, min_strength_8=0.25,
                )
                row.update(weekly_vfi_setup)
                row["VFI_1W_Value"] = float(valid_vfi["VFI"].iloc[-1])
                row["VFI_1W_Candle_Date"] = valid_vfi.index[-1].strftime("%Y-%m-%d")
                row["VFI_1W_Analysis_Status"] = "ANALYZED"
            except Exception as exc:
                row["VFI_1W_Analysis_Status"] = f"ERROR: {exc!r}"
            try:
                divergence = VolumeDivergenceMM().calculate(weekly)
                valid_div = divergence.loc[divergence["Volume_Divergence_MM"].notna()]
                if valid_div.empty:
                    raise ValueError("No valid weekly volume divergence")
                weekly_div_setup = score_zero_approach(
                    valid_div["Volume_Divergence_MM"], "Divergence_1W",
                    lookback=260, min_history=100, near_zero_fraction=0.15,
                    max_abs_near_zero=None, recent_cross_fraction=0.10,
                    recent_cross_bars=3, min_strength_4=0.10, min_strength_8=0.25,
                )
                row.update(weekly_div_setup)
                row["Divergence_1W_Value"] = float(valid_div["Volume_Divergence_MM"].iloc[-1])
                row["Weekly_Candle_Date"] = valid_div.index[-1].strftime("%Y-%m-%d")
                row["Divergence_Analysis_Status"] = "ANALYZED"
            except Exception as exc:
                row["Divergence_Analysis_Status"] = f"ERROR: {exc!r}"
            early_types = {"PRE_ZERO_BREAK", "ZERO_CROSS_RECENT"}
            vfi_weekly_type = str(row.get("VFI_1W_Setup_Type", ""))
            div_weekly_type = str(row.get("Divergence_1W_Setup_Type", ""))
            long_slopes = bool(row.get("VFI_1W_Long_Trend_20B_Positive", False)) and bool(row.get("VFI_1W_Long_Trend_50B_Positive", False))
            near_negative_turn = bool(row.get("VFI_1W_Value", 1) < 0) and bool(row.get("VFI_1W_Near_Zero", False)) and long_slopes
            weekly_candidate = vfi_weekly_type in early_types or div_weekly_type in early_types or near_negative_turn
            row["Weekly_Screen_Candidate"] = bool(weekly_candidate)
            row["Price"] = float(daily["Close"].iloc[-1]) if not daily.empty else np.nan
            row["Daily_Candle_Date"] = daily.index[-1].strftime("%Y-%m-%d") if not daily.empty else ""
            row["Weekly_Candle_Date"] = row.get("Weekly_Candle_Date") or (weekly.index[-1].strftime("%Y-%m-%d") if not weekly.empty else "")
            row["Price_Return_20B"] = float(daily["Close"].iloc[-1] / daily["Close"].iloc[-21] - 1.0) if len(daily) >= 21 else np.nan
            row["Price_Consolidation_Score"] = (
                float(np.clip(100 - max(row["Price_Return_20B"] * 100, 0) * 6 - abs(min(row["Price_Return_20B"] * 100, 0)) * 3, 0, 100))
                if np.isfinite(row["Price_Return_20B"]) else np.nan
            )

            # STAGE 2: only candidates that survive the weekly screen receive
            # the more detailed daily VFI + flow-indicator study.
            if weekly_candidate:
                daily_row = analyze_vfi_divergence(symbol, daily, weekly)
                row.update(daily_row)
                # Preserve the independent weekly-first fields if names overlap.
                row.update(weekly_vfi_setup)
                row.update(weekly_div_setup)
                flow_row = analyze_volume_indicators(symbol, daily)
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
            else:
                row["Early_Accumulation_Score"] = early_score(row)
                row["Early_Accumulation_Candidate"] = False
                row["Daily_Flow_Status"] = "NOT_RUN_WEEKLY_FILTER"
                row["Weekly_Flow_Status"] = "NOT_RUN_WEEKLY_FILTER"
                row["Avg_Dollar_Volume_20D_USD"] = as_float((daily["Close"] * daily["Volume"]).tail(20).mean()) if len(daily) >= 20 else np.nan
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
        ["Weekly_Screen_Candidate", "Early_Accumulation_Candidate", "Early_Accumulation_Score", "Avg_Dollar_Volume_20D_USD"],
        ascending=[False, False, False, False],
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
        "weekly_screen_candidates": int(result.get("Weekly_Screen_Candidate", pd.Series(dtype=bool)).fillna(False).sum()),
        "daily_followup_analyzed": int((result.get("Daily_Flow_Status", pd.Series(dtype=str)) == "AVAILABLE").sum()),
        "early_accumulation_candidates": int(result.get("Early_Accumulation_Candidate", pd.Series(dtype=bool)).fillna(False).sum()),
        "ranking": "Two-stage screen: completed weekly VFI + weekly volume divergence, including 20/50-week slopes, selects candidates; detailed daily VFI and OBV/A-D/CMF/MFI analysis runs only on weekly candidates. Final early-accumulation candidates additionally require daily/weekly flow >=50 and 20-day price return <=5%.",
        "data_provider": "Yahoo Finance via yfinance",
        "warning": "Volume-based indicators are price/volume proxies and do not identify institutional orders or guarantee future price increases.",
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    columns = [
        "Rank", "Symbol", "Analysis_Status", "Weekly_Screen_Candidate", "VFI_1W_Value",
        "VFI_1W_Slope_4B", "VFI_1W_Slope_8B", "VFI_1W_Slope_20B", "VFI_1W_Slope_50B",
        "VFI_1W_Setup_Type", "Divergence_1W_Slope_20B", "Divergence_1W_Slope_50B",
        "Early_Accumulation_Candidate",
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
