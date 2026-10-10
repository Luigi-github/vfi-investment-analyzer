"""Screen Binance assets for early VFI accumulation and weekly volume-divergence zero breaks.

Timeframes are intentionally limited to 1D for VFI and 1W for Volume Divergence
by MM. No 1H, 4H, minute, or other short-timeframe signals are used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.vfi_analyzer import VFIAnalyzer
from app.data.binance_market_data import BinanceMarketDataService
from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM
from app.scoring.vfi_score import VFIScore
from app.research.accumulation_breakout_screen import (
    rank_accumulation_candidates,
    score_zero_approach,
)


def ranked_common_universe(service: BinanceMarketDataService, top: int, min_volume: float) -> list[str]:
    spot = service.discover_spot_usdt(min_volume)
    perp = service.discover_usdt_perpetuals(min_volume)
    spot_rank = {symbol: rank for rank, symbol in enumerate(spot)}
    perp_rank = {symbol: rank for rank, symbol in enumerate(perp)}
    common = set(spot_rank) & set(perp_rank)
    return sorted(
        common,
        key=lambda symbol: (
            spot_rank[symbol] + perp_rank[symbol],
            max(spot_rank[symbol], perp_rank[symbol]),
            symbol,
        ),
    )[:top]


def analyze_symbol(
    symbol: str,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    vfi_indicator: VFIIndicator | None = None,
    divergence_indicator: VolumeDivergenceMM | None = None,
) -> dict:
    if daily is None or daily.empty:
        raise ValueError("No daily Spot OHLCV data")
    if weekly is None or weekly.empty:
        raise ValueError("No weekly Spot OHLCV data")

    vfi_indicator = vfi_indicator or VFIIndicator()
    divergence_indicator = divergence_indicator or VolumeDivergenceMM()

    # The VFI is deliberately calculated on DAILY candles (not hourly candles).
    vfi_raw = vfi_indicator.calculate(daily)
    vfi_analyzed = VFIAnalyzer().analyze(vfi_raw)
    vfi_scored = VFIScore().calculate(vfi_analyzed)
    vfi_valid = vfi_scored.loc[vfi_scored["VFI"].notna()]
    if vfi_valid.empty:
        raise ValueError(
            f"Insufficient daily history for VFI warmup: daily_rows={len(daily)}"
        )
    vfi_latest = vfi_valid.iloc[-1]

    # TradingView's Volume Divergence by MM is evaluated on WEEKLY candles,
    # roughly matching the slower context the user compares with daily VFI.
    divergence = divergence_indicator.calculate(weekly)
    div_valid = divergence.loc[divergence["Volume_Divergence_MM"].notna()]
    if div_valid.empty:
        raise ValueError(
            f"Insufficient weekly history for Volume Divergence warmup: weekly_rows={len(weekly)}"
        )
    div_latest = div_valid.iloc[-1]

    vfi_setup = score_zero_approach(
        vfi_valid["VFI"],
        "VFI_1D",
        lookback=52,
        min_history=20,
        near_zero_fraction=0.15,
        max_abs_near_zero=2.0,
        recent_cross_fraction=0.10,
        min_strength_4=0.10,
        min_strength_8=0.25,
    )
    vfi_stage = str(vfi_latest.get("VFI_Cycle_Stage", "UNKNOWN"))
    vfi_stage_eligible = vfi_stage in {
        "EARLY_RECOVERY", "SUSTAINED_RECOVERY", "APPROACHING_ZERO",
        "ZERO_CROSS_UP", "ACCUMULATION",
    }
    # Require the existing VFI cycle classifier to agree with a recovery/
    # accumulation phase. A slope-only oscillator outside those phases is
    # retained for audit but cannot become a VFI candidate.
    if not vfi_stage_eligible and vfi_setup["VFI_1D_Setup_Type"] in {
        "PRE_ZERO_BREAK", "ZERO_CROSS_RECENT",
    }:
        vfi_setup["VFI_1D_Setup_Type"] = "NONE"
        vfi_setup["VFI_1D_Setup_Score"] = min(
            float(vfi_setup["VFI_1D_Setup_Score"]), 59.9
        )
    divergence_setup = score_zero_approach(
        div_valid["Volume_Divergence_MM"],
        "Divergence_1W",
        lookback=52,
        min_history=20,
        near_zero_fraction=0.15,
        max_abs_near_zero=None,
        recent_cross_fraction=0.10,
        min_strength_4=0.10,
        min_strength_8=0.25,
    )

    bull_age = div_latest.get("Volume_Div_Bullish_Age_Bars", np.nan)
    bear_age = div_latest.get("Volume_Div_Bearish_Age_Bars", np.nan)
    bullish_recent = pd.notna(bull_age) and float(bull_age) <= 8
    bearish_recent = pd.notna(bear_age) and float(bear_age) <= 8

    row = {
        "Symbol": symbol,
        "Price": float(vfi_latest["Close"]),
        "Daily_Candle_Date": vfi_valid.index[-1].strftime("%Y-%m-%d"),
        "Weekly_Candle_Date": div_valid.index[-1].strftime("%Y-%m-%d"),
        "Daily_Bars": int(len(daily)),
        "Weekly_Bars": int(len(weekly)),
        "VFI_State_1D": str(vfi_latest.get("VFI_State", "UNKNOWN")),
        "VFI_Cycle_Stage_1D": vfi_stage,
        "VFI_Stage_Eligible_1D": bool(vfi_stage_eligible),
        "VFI_Recovery_Score_1D": float(vfi_latest.get("VFI_Recovery_Score", np.nan)),
        "VFI_Accumulation_Confirmed_1D": bool(vfi_latest.get("Accumulation_Confirmed", False)),
        "Divergence_MM_Signal_1W": str(div_latest.get("Volume_Div_Signal", "")),
        "Bullish_Divergence_Recent": bool(bullish_recent),
        "Bullish_Divergence_Age_Weeks": float(bull_age) if pd.notna(bull_age) else np.nan,
        "Bearish_Divergence_Recent": bool(bearish_recent),
        "Bearish_Divergence_Age_Weeks": float(bear_age) if pd.notna(bear_age) else np.nan,
    }
    row.update(vfi_setup)
    row.update(divergence_setup)
    # Stable names consumed by the ranking helper; detailed columns retain
    # their timeframe suffixes for auditability.
    row["VFI_Setup_Type"] = vfi_setup["VFI_1D_Setup_Type"]
    row["VFI_Setup_Score"] = vfi_setup["VFI_1D_Setup_Score"]
    row["Divergence_Setup_Type"] = divergence_setup["Divergence_1W_Setup_Type"]
    row["Divergence_Setup_Score"] = divergence_setup["Divergence_1W_Setup_Score"]
    ranked = rank_accumulation_candidates(pd.DataFrame([row]))
    return ranked.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily VFI + weekly Volume Divergence early zero-break screen (no intraday timeframes)."
    )
    parser.add_argument("--symbols", default="AUTO", help="AUTO or comma-separated Binance symbols")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--min-volume", type=float, default=0.0)
    parser.add_argument("--daily-bars", type=int, default=1000)
    parser.add_argument("--weekly-bars", type=int, default=400)
    parser.add_argument("--output-dir", default="reports/crypto_accumulation_breakout")
    args = parser.parse_args()

    if args.top < 1 or args.daily_bars < 160 or args.weekly_bars < 100:
        parser.error("--top must be positive; use at least 160 daily bars and 100 weekly bars")

    service = BinanceMarketDataService()
    if args.symbols.strip().upper() == "AUTO":
        symbols = ranked_common_universe(service, args.top, args.min_volume)
    else:
        symbols = list(dict.fromkeys(
            item.strip().upper() for item in args.symbols.split(",") if item.strip()
        ))[:args.top]
    if not symbols:
        raise RuntimeError("No Binance Spot + USD-M perpetual symbols found")

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []

    for number, symbol in enumerate(symbols, start=1):
        print(f"[{number}/{len(symbols)}] {symbol}", flush=True)
        try:
            daily = service.get_klines(symbol, "1d", market="spot", max_bars=args.daily_bars)
            weekly = service.get_klines(symbol, "1w", market="spot", max_bars=args.weekly_bars)
            row = analyze_symbol(symbol, daily, weekly)
            row["Universe_Liquidity_Rank"] = number
            rows.append(row)
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": repr(exc)})
            print(f"ERROR {symbol}: {exc}", flush=True)

    result = rank_accumulation_candidates(pd.DataFrame(rows))
    if not result.empty:
        result.insert(0, "Rank", range(1, len(result) + 1))
        vfi_candidates = result[result["VFI_Setup"]].copy()
        combined_candidates = result[result["Combined_Setup"]].copy()
        result.to_csv(output / "all_assets.csv", index=False)
        vfi_candidates.to_csv(output / "vfi_daily_candidates.csv", index=False)
        combined_candidates.to_csv(output / "combined_vfi_divergence_candidates.csv", index=False)
    else:
        vfi_candidates = result.copy()
        combined_candidates = result.copy()
        result.to_csv(output / "all_assets.csv", index=False)
        vfi_candidates.to_csv(output / "vfi_daily_candidates.csv", index=False)
        combined_candidates.to_csv(output / "combined_vfi_divergence_candidates.csv", index=False)

    pd.DataFrame(errors, columns=["Symbol", "Error"]).to_csv(output / "errors.csv", index=False)
    summary = {
        "universe_requested": len(symbols),
        "assets_analyzed": len(result),
        "errors": len(errors),
        "vfi_daily_candidates": int(len(vfi_candidates)),
        "combined_daily_vfi_weekly_divergence_candidates": int(len(combined_candidates)),
        "timeframes": {"VFI": "1D", "Volume_Divergence_by_MM": "1W"},
        "intraday_timeframes_used": [],
        "ranking_logic": (
            "Tier 1: both daily VFI and weekly divergence are negative and near zero "
            "with positive 1/4/8-bar slopes (or a very recent, small zero cross). "
            "Tier 2: daily VFI setup only. Remaining assets follow by diagnostic score. "
            "Recent bullish pivot divergence is a separate confirmation bonus."
        ),
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not result.empty:
        display_columns = [
            "Rank", "Symbol", "Setup_Tier", "Combined_Setup", "Combined_Opportunity_Score",
            "VFI_1D_Value", "VFI_1D_Slope_1B", "VFI_1D_Slope_4B", "VFI_1D_Slope_8B",
            "VFI_1D_Setup_Type", "Divergence_1W_Value", "Divergence_1W_Slope_1B",
            "Divergence_1W_Slope_4B", "Divergence_1W_Slope_8B",
            "Divergence_1W_Setup_Type", "Bullish_Divergence_Recent",
        ]
        print(result[[col for col in display_columns if col in result.columns]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
