"""Two-stage US equity/ETF accumulation screen: weekly universe -> daily shortlist."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM
from scripts.run_volume_accumulation_indicators import calculate_indicators, completed_daily, to_weekly
from scripts.run_us_top100_liquid_screen import US_CANDIDATE_UNIVERSE

# Combine a broad curated stock universe with heavily traded ETFs, then rank
# the full seed universe by 20-session dollar turnover before weekly analysis.
US_LIQUID_UNIVERSE = list(dict.fromkeys("""
AAPL MSFT NVDA AMZN META GOOGL GOOG TSLA AVGO AMD INTC QCOM MU ORCL CRM PLTR NFLX
JPM BAC WFC C GS MS V MA AXP XOM CVX COP SLB GE CAT BA HON RTX LMT UNH LLY JNJ ABBV MRK PFE AMGN COST WMT TGT HD LOW MCD KO PEP PG
DIS CMCSA T VZ TMUS UBER ABNB SHOP COIN HOOD XYZ PYPL SOFI RIVN F FDX UPS NKE SBUX TSM ASML ARM SMCI LULU CRWD PANW NOW IBM CSCO TXN
SPY QQQ IWM DIA TLT GLD SLV XLF XLK XLE XLV SMH SOXX HYG LQD EEM EFA IBIT
""".split() + US_CANDIDATE_UNIVERSE))

def safe_slope(series, bars=4):
    s = pd.to_numeric(series, errors="coerce").dropna().tail(bars)
    if len(s) < bars:
        return float("nan")
    return float(np.polyfit(np.arange(len(s), dtype=float), s.to_numpy(dtype=float), 1)[0])


def vfi_snapshot(data):
    """VFI-only snapshot; slopes are linear-regression slopes over completed bars."""
    try:
        v = VFIIndicator().calculate(data)
        valid = v.loc[v["VFI"].notna()]
        if len(valid) < 50:
            return {}
        s = pd.to_numeric(valid["VFI"], errors="coerce").dropna()
        latest = valid.loc[s.index[-1]]
        last_vfi = float(s.iloc[-1])
        return {
            "VFI": last_vfi,
            "VFI_Prev": float(s.iloc[-2]) if len(s) >= 2 else float("nan"),
            "VFI_Signal": float(latest.get("VFI_Signal", np.nan)),
            "VFI_Histogram": float(latest.get("VFI_Histogram", np.nan)),
            "VFI_Slope_4B": safe_slope(s, 4),
            "VFI_Slope_8B": safe_slope(s, 8),
            "VFI_Slope_20B": safe_slope(s, 20),
            "VFI_Slope_50B": safe_slope(s, 50),
            "VFI_Zero_Cross_Recent_4B": bool(
                len(s.tail(4)) >= 2 and
                ((s.tail(4).shift(1) < 0) & (s.tail(4) >= 0)).any()
            ),
        }
    except Exception as exc:
        return {"VFI_Error": repr(exc)}


def weekly_screen(symbol, daily):
    daily = completed_daily(daily)
    weekly = to_weekly(daily)
    if len(weekly) < 60:
        raise ValueError(f"Only {len(weekly)} completed weekly bars; need 60 for the 50-week slope")
    vfi = vfi_snapshot(weekly)
    if not vfi or "VFI" not in vfi:
        raise ValueError(f"Unable to calculate weekly VFI: {vfi.get('VFI_Error', 'insufficient VFI data')}")
    value = vfi["VFI"]
    slope20 = vfi["VFI_Slope_20B"]
    slope50 = vfi["VFI_Slope_50B"]
    # Require a clean recent rise, but avoid demanding eight flawless weeks,
    # which produced zero candidates in the first 100-liquid-asset run.
    vfi_series = pd.to_numeric(VFIIndicator().calculate(weekly)["VFI"], errors="coerce").dropna()
    recent4 = vfi_series.tail(4).to_numpy(dtype=float)
    deltas4 = np.diff(recent4)
    monotonic4 = bool(len(recent4) == 4 and np.all(deltas4 > 0))
    x4 = np.arange(len(recent4), dtype=float)
    fit4 = np.polyval(np.polyfit(x4, recent4, 1), x4) if len(recent4) == 4 else np.array([])
    ss_total4 = float(np.sum((recent4 - recent4.mean()) ** 2)) if len(recent4) == 4 else 0.0
    r2_4 = float(1 - np.sum((recent4 - fit4) ** 2) / ss_total4) if ss_total4 > 0 else 0.0
    slope8 = vfi["VFI_Slope_8B"]
    price_return_13w = float(weekly["Close"].iloc[-1] / weekly["Close"].iloc[-14] - 1)
    # Early accumulation: VFI remains below zero, recent VFI rises each of the
    # last 4 completed weeks in a line-like path, the 8W/20W trends rise, and
    # price has not already advanced more than 10% over 13 weeks.
    qualifies = bool(
        np.isfinite(value) and value < 0
        and np.isfinite(slope8) and slope8 > 0
        and np.isfinite(slope20) and slope20 > 0
        and monotonic4 and r2_4 >= 0.65
        and np.isfinite(price_return_13w) and price_return_13w <= 0.10
    )
    # Rank qualifying names by closeness to zero, then by longer/shorter slope.
    score = float(100.0 / (1.0 + abs(value))) if np.isfinite(value) else 0.0
    return {
        "Symbol": symbol,
        "Weekly_Screen_Status": "CANDIDATE" if qualifies else "NOT_SELECTED",
        "Weekly_Score": score,
        "Weekly_VFI": value,
        "Weekly_VFI_Prev": vfi.get("VFI_Prev"),
        "Weekly_VFI_Signal": vfi.get("VFI_Signal"),
        "Weekly_VFI_Histogram": vfi.get("VFI_Histogram"),
        "Weekly_VFI_Last8_Monotonic_Rise": monotonic8,
        "Weekly_VFI_R2_8W": r2_8,
        "Weekly_VFI_Slope_20W": slope20,
        "Weekly_VFI_Slope_50W": slope50,
        "Weekly_VFI_Slope_4W": vfi.get("VFI_Slope_4B"),
        "Weekly_VFI_Slope_8W": vfi.get("VFI_Slope_8B"),
        "Weekly_Candle_Date": weekly.index[-1].strftime("%Y-%m-%d"),
        "Weekly_Price": float(weekly["Close"].iloc[-1]),
        "Weekly_Price_Return_13W": float(weekly["Close"].iloc[-1] / weekly["Close"].iloc[-14] - 1),
        "_daily": daily,
        "_weekly": weekly,
        "_qualifies": qualifies,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Weekly-only screen for early accumulation using a smooth rising VFI.")
    p.add_argument("--symbols", default=",".join(US_LIQUID_UNIVERSE))
    p.add_argument("--period", default="10y")
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--max-daily-candidates", type=int, default=50)
    p.add_argument("--output-dir", default="reports/us_weekly_then_daily")
    args = p.parse_args()
    symbols = list(dict.fromkeys(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    service = MarketDataService()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_data, liquidity_rows, errors = {}, [], []
    print(f"Downloading daily history for {len(symbols)} US-listed symbols to rank liquidity...", flush=True)
    for i, symbol in enumerate(symbols, 1):
        print(f"[LIQUIDITY {i}/{len(symbols)}] {symbol}", flush=True)
        try:
            daily = completed_daily(service.get_daily_data(symbol, period=args.period))
            if daily.empty or not {"Close", "Volume"}.issubset(daily.columns):
                raise ValueError("No usable OHLCV data")
            raw_data[symbol] = daily
            dollar_turnover = (
                pd.to_numeric(daily["Close"], errors="coerce") *
                pd.to_numeric(daily["Volume"], errors="coerce")
            ).replace([np.inf, -np.inf], np.nan).dropna().tail(20)
            turnover = float(dollar_turnover.mean()) if len(dollar_turnover) >= 10 else float("nan")
            liquidity_rows.append({
                "Symbol": symbol, "Average_Dollar_Turnover_20D": turnover,
                "Liquidity_Data_Date": daily.index[-1].strftime("%Y-%m-%d"),
                "Liquidity_Status": "OK" if np.isfinite(turnover) else "INSUFFICIENT_DATA",
            })
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "LIQUIDITY", "Error": repr(exc)})
            liquidity_rows.append({"Symbol": symbol, "Average_Dollar_Turnover_20D": np.nan,
                                   "Liquidity_Data_Date": "", "Liquidity_Status": f"ERROR: {exc!r}"})

    liquidity_df = pd.DataFrame(liquidity_rows)
    eligible = liquidity_df[np.isfinite(pd.to_numeric(liquidity_df["Average_Dollar_Turnover_20D"], errors="coerce"))]
    selected = eligible.sort_values("Average_Dollar_Turnover_20D", ascending=False).head(args.top).copy()
    selected_symbols = selected["Symbol"].tolist()
    liquidity_rank = {r.Symbol: (rank, float(r.Average_Dollar_Turnover_20D))
                      for rank, r in enumerate(selected.itertuples(index=False), 1)}
    print(f"Selected {len(selected_symbols)} assets by 20-session dollar turnover.", flush=True)

    weekly_rows, frames = [], {}
    for i, symbol in enumerate(selected_symbols, 1):
        print(f"[WEEKLY VFI {i}/{len(selected_symbols)}] {symbol}", flush=True)
        try:
            row = weekly_screen(symbol, raw_data[symbol])
            frames[symbol] = (row.pop("_daily"), row.pop("_weekly"))
            row["_qualifies"] = bool(row.pop("_qualifies"))
            row["Liquidity_Rank"], row["Average_Dollar_Turnover_20D"] = liquidity_rank[symbol]
            weekly_rows.append(row)
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "WEEKLY_VFI", "Error": repr(exc)})
            weekly_rows.append({"Symbol": symbol, "Weekly_Screen_Status": "ERROR",
                                "Error": repr(exc), "Liquidity_Rank": liquidity_rank[symbol][0],
                                "Average_Dollar_Turnover_20D": liquidity_rank[symbol][1]})
    weekly_df = pd.DataFrame(weekly_rows)
    if "_qualifies" in weekly_df:
        weekly_df = weekly_df.drop(columns=["_qualifies"])
    weekly_df.to_csv(out / "us_100_weekly_screen.csv", index=False)
    liquidity_df.sort_values("Average_Dollar_Turnover_20D", ascending=False).to_csv(out / "us_liquidity_universe.csv", index=False)

    candidates = weekly_df[weekly_df["Weekly_Screen_Status"].eq("CANDIDATE")].copy()
    for col in ["Weekly_VFI", "Weekly_VFI_Slope_20W", "Weekly_VFI_Slope_50W", "Weekly_Score"]:
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    candidates["VFI_Distance_To_Zero"] = candidates["Weekly_VFI"].abs()
    candidates = candidates.sort_values(
        ["VFI_Distance_To_Zero", "Weekly_VFI_R2_8W", "Weekly_VFI_Slope_50W", "Weekly_VFI_Slope_20W"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    candidates.insert(0, "Rank", range(1, len(candidates) + 1))
    candidates.to_csv(out / "us_weekly_vfi_candidates.csv", index=False)
    print("\nWEEKLY VFI CANDIDATES: VFI < 0 and positive 20W + 50W slopes", flush=True)
    print(candidates.to_string(index=False) if not candidates.empty else "No assets passed the exact weekly VFI filter.", flush=True)

    # Deliberately no daily technical-indicator calculations; stop after weekly ranking.

    summary = {
        "seed_universe_size": len(symbols), "requested_top_liquid": args.top,
        "selected_by_20_session_dollar_turnover": len(selected_symbols),
        "weekly_analyzed": int(weekly_df["Weekly_Screen_Status"].ne("ERROR").sum()) if not weekly_df.empty else 0,
        "weekly_vfi_candidates": int(len(candidates)),
        "weekly_rule": "completed weekly VFI < 0; 8W and 20W slopes > 0; last 4 weekly VFI values strictly increase; 4W linear-fit R² >= 0.65; 13W price return <= +10%",
        "daily_rule": "not calculated; daily OHLCV is used only to rank dollar liquidity",
        "weekly_candle_policy": "completed weekly candles only; incomplete current week excluded",
        "universe_note": "Top N by mean Close*Volume over 20 completed sessions within a curated US-listed seed universe; not all US listings.",
        "method_note": "VFI-only selection. OHLCV-based volume indicator; not direct institutional-flow data.",
        "errors": errors,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nRUN SUMMARY\n" + json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
