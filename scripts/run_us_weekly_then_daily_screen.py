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
    # The only weekly selection rule: VFI is below zero and both 20- and
    # 50-week linear-regression slopes are positive. No other indicators gate.
    qualifies = bool(
        np.isfinite(value) and value < 0
        and np.isfinite(slope20) and slope20 > 0
        and np.isfinite(slope50) and slope50 > 0
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
    p = argparse.ArgumentParser(description="Screen top-liquid US assets by weekly VFI below zero and positive 20/50-week slopes, then inspect daily VFI.")
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
        ["VFI_Distance_To_Zero", "Weekly_VFI_Slope_50W", "Weekly_VFI_Slope_20W"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    candidates.insert(0, "Rank", range(1, len(candidates) + 1))
    candidates.to_csv(out / "us_weekly_vfi_candidates.csv", index=False)
    print("\nWEEKLY VFI CANDIDATES: VFI < 0 and positive 20W + 50W slopes", flush=True)
    print(candidates.to_string(index=False) if not candidates.empty else "No assets passed the exact weekly VFI filter.", flush=True)

    chosen = candidates["Symbol"].head(args.max_daily_candidates).tolist()
    daily_rows = []
    for i, symbol in enumerate(chosen, 1):
        print(f"[DAILY VFI {i}/{len(chosen)}] {symbol}", flush=True)
        daily, weekly = frames[symbol]
        try:
            dvfi = vfi_snapshot(daily)
            if "VFI" not in dvfi:
                raise ValueError(dvfi.get("VFI_Error", "Daily VFI unavailable"))
            daily_rows.append({
                "Symbol": symbol,
                "Weekly_Rank": int(candidates.loc[candidates["Symbol"].eq(symbol), "Rank"].iloc[0]),
                "Weekly_VFI": float(candidates.loc[candidates["Symbol"].eq(symbol), "Weekly_VFI"].iloc[0]),
                "Weekly_VFI_Slope_20W": float(candidates.loc[candidates["Symbol"].eq(symbol), "Weekly_VFI_Slope_20W"].iloc[0]),
                "Weekly_VFI_Slope_50W": float(candidates.loc[candidates["Symbol"].eq(symbol), "Weekly_VFI_Slope_50W"].iloc[0]),
                "Weekly_Candle_Date": weekly.index[-1].strftime("%Y-%m-%d"),
                "Daily_Candle_Date": daily.index[-1].strftime("%Y-%m-%d"),
                "Daily_Price": float(daily["Close"].iloc[-1]),
                "Daily_Price_Return_20B": float(daily["Close"].iloc[-1] / daily["Close"].iloc[-21] - 1) if len(daily) >= 21 else float("nan"),
                "Daily_VFI": dvfi["VFI"],
                "Daily_VFI_Slope_20D": dvfi["VFI_Slope_20B"],
                "Daily_VFI_Slope_50D": dvfi["VFI_Slope_50B"],
                "Daily_VFI_Slope_4D": dvfi["VFI_Slope_4B"],
                "Daily_VFI_Slope_8D": dvfi["VFI_Slope_8B"],
                "Daily_VFI_Zero_Cross_Recent_4D": dvfi["VFI_Zero_Cross_Recent_4B"],
                "Daily_Interpretation": (
                    "DAILY_VFI_ALSO_BELOW_ZERO_AND_RISING_20_50D"
                    if dvfi["VFI"] < 0 and dvfi["VFI_Slope_20B"] > 0 and dvfi["VFI_Slope_50B"] > 0
                    else "WEEKLY_FILTER_PASSED_DAILY_VFI_NEEDS_REVIEW"
                ),
            })
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "DAILY_VFI", "Error": repr(exc)})
    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df = daily_df.sort_values(
            ["Daily_Interpretation", "Weekly_Rank"], ascending=[True, True]
        ).reset_index(drop=True)
        daily_df.insert(0, "Daily_Rank", range(1, len(daily_df) + 1))
    daily_df.to_csv(out / "us_daily_shortlist_ranking.csv", index=False)
    strict_daily = daily_df[daily_df["Daily_Interpretation"].eq("DAILY_VFI_ALSO_BELOW_ZERO_AND_RISING_20_50D")].copy() if not daily_df.empty else daily_df.copy()
    strict_daily.to_csv(out / "us_daily_strict_vfi_candidates.csv", index=False)
    summary = {
        "seed_universe_size": len(symbols), "requested_top_liquid": args.top,
        "selected_by_20_session_dollar_turnover": len(selected_symbols),
        "weekly_analyzed": int(weekly_df["Weekly_Screen_Status"].ne("ERROR").sum()) if not weekly_df.empty else 0,
        "weekly_vfi_candidates": int(len(candidates)), "daily_analyzed": int(len(daily_df)),
        "daily_also_meets_vfi_20_50_slope_filter": int(len(strict_daily)),
        "weekly_rule": "completed weekly VFI < 0; linear-regression slopes over the latest 20 and 50 completed weekly VFI values both > 0",
        "daily_rule": "daily review only for weekly-qualified assets; separately flags daily VFI < 0 with positive 20- and 50-day slopes",
        "weekly_candle_policy": "completed weekly candles only",
        "daily_candle_policy": "completed daily candles only",
        "universe_note": "Top N by mean Close*Volume over 20 completed sessions within a curated US-listed seed universe; not all US listings.",
        "method_note": "VFI-only selection. OHLCV-based volume indicator; not direct institutional-flow data.",
        "errors": errors,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nRUN SUMMARY\n" + json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
