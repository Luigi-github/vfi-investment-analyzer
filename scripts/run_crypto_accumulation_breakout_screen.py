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
    unavailable_asset_row,
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


def universe_liquidity_metadata(
    service: BinanceMarketDataService,
    symbols: list[str],
) -> dict[str, dict]:
    """Attach auditable 24-hour quote volume and full-universe ranks to selected symbols."""
    try:
        spot_tickers = service._get(service.SPOT_BASE, "/api/v3/ticker/24hr", {})
        spot_error = ""
    except Exception as exc:
        spot_tickers, spot_error = [], repr(exc)
    try:
        perp_tickers = service._get(service.FUTURES_BASE, "/fapi/v1/ticker/24hr", {})
        perp_error = ""
    except Exception as exc:
        perp_tickers, perp_error = [], repr(exc)

    def volume_map(rows):
        result = {}
        for item in rows if isinstance(rows, list) else []:
            try:
                result[str(item["symbol"])] = float(item.get("quoteVolume", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
        return result

    spot_volume = volume_map(spot_tickers)
    perp_volume = volume_map(perp_tickers)
    spot_rank = {
        symbol: rank
        for rank, (symbol, _) in enumerate(
            sorted(spot_volume.items(), key=lambda item: (-item[1], item[0])), start=1
        )
    }
    perp_rank = {
        symbol: rank
        for rank, (symbol, _) in enumerate(
            sorted(perp_volume.items(), key=lambda item: (-item[1], item[0])), start=1
        )
    }
    return {
        symbol: {
            "Spot_Quote_Volume_24H_USDT": spot_volume.get(symbol, np.nan),
            "Perpetual_Quote_Volume_24H_USDT": perp_volume.get(symbol, np.nan),
            "Spot_Quote_Volume_Rank": spot_rank.get(symbol, np.nan),
            "Perpetual_Quote_Volume_Rank": perp_rank.get(symbol, np.nan),
            "Combined_Liquidity_Rank_Sum": (
                spot_rank[symbol] + perp_rank[symbol]
                if symbol in spot_rank and symbol in perp_rank else np.nan
            ),
            "Spot_Liquidity_Metadata_Status": (
                "AVAILABLE" if symbol in spot_volume else ("FETCH_ERROR" if spot_error else "MISSING_SYMBOL")
            ),
            "Perpetual_Liquidity_Metadata_Status": (
                "AVAILABLE" if symbol in perp_volume else ("FETCH_ERROR" if perp_error else "MISSING_SYMBOL")
            ),
            "Liquidity_Metadata_Error": "; ".join(
                error for error in [spot_error, perp_error] if error
            ),
        }
        for symbol in symbols
    }


def analyze_symbol(
    symbol: str,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    vfi_indicator: VFIIndicator | None = None,
    divergence_indicator: VolumeDivergenceMM | None = None,
) -> dict:
    """Analyze VFI and divergence independently so one short history cannot erase the other signal."""
    vfi_indicator = vfi_indicator or VFIIndicator(context_window=365)
    divergence_indicator = divergence_indicator or VolumeDivergenceMM()

    vfi_setup = score_zero_approach(
        pd.Series(dtype=float), "VFI_1D", lookback=365, min_history=365
    )
    divergence_setup = score_zero_approach(
        pd.Series(dtype=float), "Divergence_1W", lookback=52, min_history=20
    )
    vfi_latest = None
    div_latest = None
    vfi_reason = ""
    div_reason = ""
    vfi_valid = pd.DataFrame()
    div_valid = pd.DataFrame()

    # VFI uses daily candles. Keep its result even if weekly divergence history is too short.
    try:
        if daily is None or daily.empty:
            raise ValueError("No daily Spot OHLCV data")
        vfi_raw = vfi_indicator.calculate(daily)
        vfi_analyzed = VFIAnalyzer().analyze(vfi_raw)
        vfi_scored = VFIScore().calculate(vfi_analyzed)
        vfi_valid = vfi_scored.loc[vfi_scored["VFI"].notna()]
        if vfi_valid.empty:
            raise ValueError(
                f"Insufficient daily history for VFI warmup: daily_rows={len(daily)}"
            )
        vfi_latest = vfi_valid.iloc[-1]
        # Compare daily VFI with the 52-week divergence context using an
        # approximately one-year daily range, not a short 52-day range.
        vfi_setup = score_zero_approach(
            vfi_valid["VFI"], "VFI_1D", lookback=365, min_history=365,
            near_zero_fraction=0.15, max_abs_near_zero=2.0,
            recent_cross_fraction=0.10, recent_cross_bars=3,
            min_strength_4=0.10, min_strength_8=0.25,
        )
        vfi_stage = str(vfi_latest.get("VFI_Cycle_Stage", "UNKNOWN"))
        # The generic VFIAnalyzer stage uses fixed raw 4/8-bar thresholds and
        # can contradict this screen's normalized 1D zero-approach criteria.
        # Keep that stage as diagnostic metadata; eligibility is decided only
        # by the explicit, timeframe-specific setup rules above.
        vfi_stage_eligible = vfi_setup["VFI_1D_Setup_Type"] in {
            "PRE_ZERO_BREAK", "ZERO_CROSS_RECENT",
        }
    except Exception as exc:
        vfi_reason = repr(exc)
        vfi_stage = "UNKNOWN"
        vfi_stage_eligible = False

    # Volume Divergence by MM uses weekly candles to match the user's TradingView comparison.
    try:
        if weekly is None or weekly.empty:
            raise ValueError("No weekly Spot OHLCV data")
        divergence = divergence_indicator.calculate(weekly)
        div_valid = divergence.loc[divergence["Volume_Divergence_MM"].notna()]
        if div_valid.empty:
            raise ValueError(
                "Insufficient weekly history for Volume Divergence warmup: "
                f"weekly_rows={len(weekly)}"
            )
        div_latest = div_valid.iloc[-1]
        divergence_setup = score_zero_approach(
            div_valid["Volume_Divergence_MM"], "Divergence_1W",
            lookback=52, min_history=20, near_zero_fraction=0.15,
            max_abs_near_zero=None, recent_cross_fraction=0.10,
            recent_cross_bars=2, min_strength_4=0.10, min_strength_8=0.25,
        )
    except Exception as exc:
        div_reason = repr(exc)

    vfi_available = vfi_latest is not None
    div_available = div_latest is not None
    if vfi_available and div_available:
        status = "ANALYZED"
    elif vfi_available or div_available:
        other_reason = div_reason if vfi_available else vfi_reason
        status = "PARTIAL_ANALYSIS" if "Insufficient" in other_reason or "No " in other_reason else "PARTIAL_ERROR"
    elif ("Insufficient" in vfi_reason or "No " in vfi_reason) and ("Insufficient" in div_reason or "No " in div_reason):
        status = "INSUFFICIENT_HISTORY"
    else:
        status = "ERROR"

    bull_age = div_latest.get("Volume_Div_Bullish_Age_Bars", np.nan) if div_available else np.nan
    bear_age = div_latest.get("Volume_Div_Bearish_Age_Bars", np.nan) if div_available else np.nan
    bullish_recent = pd.notna(bull_age) and float(bull_age) <= 8
    bearish_recent = pd.notna(bear_age) and float(bear_age) <= 8

    row = {
        "Symbol": symbol,
        "Analysis_Status": status,
        "VFI_Analysis_Status": "ANALYZED" if vfi_available else ("INSUFFICIENT_HISTORY" if "Insufficient" in vfi_reason or "No " in vfi_reason else "ERROR"),
        "Divergence_Analysis_Status": "ANALYZED" if div_available else ("INSUFFICIENT_HISTORY" if "Insufficient" in div_reason or "No " in div_reason else "ERROR"),
        "Exclusion_Reason": "; ".join(x for x in [vfi_reason, div_reason] if x),
        "Analysis_Detail": "Both indicators available" if vfi_available and div_available else "One indicator unavailable; available indicator retained",
        "Price": float(vfi_latest["Close"]) if vfi_available else np.nan,
        "Daily_Candle_Date": vfi_valid.index[-1].strftime("%Y-%m-%d") if vfi_available else "",
        "Weekly_Candle_Date": div_valid.index[-1].strftime("%Y-%m-%d") if div_available else "",
        "Daily_Bars": int(len(daily)) if daily is not None else 0,
        "Weekly_Bars": int(len(weekly)) if weekly is not None else 0,
        "VFI_State_1D": str(vfi_latest.get("VFI_State", "UNKNOWN")) if vfi_available else "UNKNOWN",
        "VFI_Cycle_Stage_1D": str(vfi_latest.get("VFI_Cycle_Stage", "UNKNOWN")) if vfi_available else "UNKNOWN",
        "VFI_Stage_Eligible_1D": bool(vfi_stage_eligible) if vfi_available else False,
        "VFI_Recovery_Score_1D": float(vfi_latest.get("VFI_Recovery_Score", np.nan)) if vfi_available else np.nan,
        "VFI_Accumulation_Confirmed_1D": bool(vfi_latest.get("Accumulation_Confirmed", False)) if vfi_available else False,
        "Divergence_MM_Signal_1W": str(div_latest.get("Volume_Div_Signal", "")) if div_available else "",
        "Bullish_Divergence_Recent": bool(bullish_recent),
        "Bullish_Divergence_Age_Weeks": float(bull_age) if pd.notna(bull_age) else np.nan,
        "Bearish_Divergence_Recent": bool(bearish_recent),
        "Bearish_Divergence_Age_Weeks": float(bear_age) if pd.notna(bear_age) else np.nan,
    }
    row.update(vfi_setup)
    row.update(divergence_setup)
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

    if args.top < 1 or args.daily_bars < 495 or args.weekly_bars < 100:
        parser.error("--top must be positive; use at least 495 daily bars (130-bar VFI warmup + 365-bar cycle context) and 100 weekly bars")

    service = BinanceMarketDataService()
    if args.symbols.strip().upper() == "AUTO":
        symbols = ranked_common_universe(service, args.top, args.min_volume)
    else:
        symbols = list(dict.fromkeys(
            item.strip().upper() for item in args.symbols.split(",") if item.strip()
        ))[:args.top]
    if not symbols:
        raise RuntimeError("No Binance Spot + USD-M perpetual symbols found")

    liquidity_metadata = universe_liquidity_metadata(service, symbols)

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []

    for number, symbol in enumerate(symbols, start=1):
        print(f"[{number}/{len(symbols)}] {symbol}", flush=True)
        daily = pd.DataFrame()
        weekly = pd.DataFrame()
        fetch_errors = []
        try:
            daily = service.get_klines(symbol, "1d", market="spot", max_bars=args.daily_bars)
        except Exception as exc:
            fetch_errors.append(f"daily fetch: {exc!r}")
        try:
            weekly = service.get_klines(symbol, "1w", market="spot", max_bars=args.weekly_bars)
        except Exception as exc:
            fetch_errors.append(f"weekly fetch: {exc!r}")

        try:
            row = analyze_symbol(symbol, daily, weekly)
            row["Universe_Liquidity_Rank"] = number
            row.update(liquidity_metadata.get(symbol, {}))
            if fetch_errors:
                row["Exclusion_Reason"] = "; ".join(
                    x for x in [str(row.get("Exclusion_Reason", "")), *fetch_errors] if x
                )
                row["Analysis_Status"] = "ERROR" if len(fetch_errors) == 2 else "PARTIAL_ERROR"
                if not len(daily):
                    row["VFI_Analysis_Status"] = "ERROR"
                if not len(weekly):
                    row["Divergence_Analysis_Status"] = "ERROR"
            rows.append(row)
            if row["Analysis_Status"] in {"ERROR", "PARTIAL_ERROR"}:
                errors.append({"Symbol": symbol, "Error": row.get("Exclusion_Reason", row["Analysis_Status"])})
                print(f"{row['Analysis_Status']} {symbol}: {row.get('Exclusion_Reason', '')}", flush=True)
            elif row["Analysis_Status"] == "PARTIAL_ANALYSIS":
                print(f"PARTIAL HISTORY {symbol}: {row.get('Exclusion_Reason', '')}", flush=True)
        except Exception as exc:
            reason = repr(exc)
            rows.append(unavailable_asset_row(symbol, number, "ERROR", reason))
            errors.append({"Symbol": symbol, "Error": reason})
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
    status_counts = (
        result["Analysis_Status"].value_counts().to_dict()
        if not result.empty and "Analysis_Status" in result else {}
    )
    summary = {
        "universe_requested": len(symbols),
        "assets_reported": int(len(result)),
        "assets_analyzed": int(status_counts.get("ANALYZED", 0)),
        "partial_analysis": int(status_counts.get("PARTIAL_ANALYSIS", 0)),
        "partial_errors": int(status_counts.get("PARTIAL_ERROR", 0)),
        "insufficient_history": int(status_counts.get("INSUFFICIENT_HISTORY", 0)),
        "errors": int(status_counts.get("ERROR", 0)),
        "vfi_daily_candidates": int(len(vfi_candidates)),
        "weekly_divergence_candidates": int(result["Divergence_Setup"].sum()) if "Divergence_Setup" in result else 0,
        "combined_daily_vfi_weekly_divergence_candidates": int(len(combined_candidates)),
        "vfi_cycle_history_insufficient": int((result["VFI_Setup_Type"] == "INSUFFICIENT_HISTORY").sum()) if "VFI_Setup_Type" in result else 0,
        "weekly_divergence_history_insufficient": int((result["Divergence_Setup_Type"] == "INSUFFICIENT_HISTORY").sum()) if "Divergence_Setup_Type" in result else 0,
        "timeframes": {"VFI": "1D", "Volume_Divergence_by_MM": "1W"},
        "intraday_timeframes_used": [],
        "ranking_logic": (
            "VFI uses completed daily candles with a 365-bar context and requires at least "
            "365 valid VFI values after indicator warm-up; weekly Volume Divergence by MM "
            "uses completed weekly candles with a 52-bar context. A PRE_ZERO_BREAK requires "
            "negative oscillator value near zero, positive 1/4/8-bar slopes, normalized "
            "4-bar rise >= 0.10 and 8-bar rise >= 0.25 standard deviations. A recent zero "
            "cross is a separate setup only if it remains within 10% of the range and passes "
            "the same slope tests. Tier 1 requires both setups; Tier 2 requires daily VFI; "
            "Tier 3 retains near misses for audit. No intraday frames are used. The row-level "
            "Setup_Reason fields explain each accept/reject decision; recent bullish pivot "
            "divergence is only an additional confirmation bonus."
        ),
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not result.empty:
        display_columns = [
            "Rank", "Symbol", "Setup_Tier", "Combined_Setup", "Combined_Opportunity_Score",
            "VFI_1D_Value", "VFI_1D_Slope_1B", "VFI_1D_Slope_4B", "VFI_1D_Slope_8B",
            "VFI_1D_Setup_Type", "VFI_1D_Setup_Reason",
            "Divergence_1W_Value", "Divergence_1W_Slope_1B",
            "Divergence_1W_Slope_4B", "Divergence_1W_Slope_8B",
            "Divergence_1W_Setup_Type", "Divergence_1W_Setup_Reason",
            "Bullish_Divergence_Recent",
        ]
        print(result[[col for col in display_columns if col in result.columns]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
