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
    # Explicitly match the user's weekly setup: VFI has spent weeks below zero
    # (negative 20W and 50W means), while the last four completed weekly values
    # rise in a near-linear path and current VFI is still close to zero or has
    # only just crossed it.
    vfi_series = pd.to_numeric(VFIIndicator().calculate(weekly)["VFI"], errors="coerce").dropna()
    vfi_ma50 = float(vfi_series.tail(50).mean()) if len(vfi_series) >= 50 else float("nan")
    vfi_ma20 = float(vfi_series.tail(20).mean()) if len(vfi_series) >= 20 else float("nan")
    recent4 = vfi_series.tail(4).to_numpy(dtype=float)
    deltas4 = np.diff(recent4)
    monotonic4 = bool(len(recent4) == 4 and np.all(deltas4 > 0))
    x4 = np.arange(len(recent4), dtype=float)
    fit4 = np.polyval(np.polyfit(x4, recent4, 1), x4) if len(recent4) == 4 else np.array([])
    ss_total4 = float(np.sum((recent4 - recent4.mean()) ** 2)) if len(recent4) == 4 else 0.0
    r2_4 = float(1 - np.sum((recent4 - fit4) ** 2) / ss_total4) if ss_total4 > 0 else 0.0
    recent8 = vfi_series.tail(8).to_numpy(dtype=float)
    x8 = np.arange(len(recent8), dtype=float)
    fit8 = np.polyval(np.polyfit(x8, recent8, 1), x8) if len(recent8) == 8 else np.array([])
    ss_total8 = float(np.sum((recent8 - recent8.mean()) ** 2)) if len(recent8) == 8 else 0.0
    r2_8 = float(1 - np.sum((recent8 - fit8) ** 2) / ss_total8) if ss_total8 > 0 else 0.0
    slope8 = vfi["VFI_Slope_8B"]
    price_return_13w = float(weekly["Close"].iloc[-1] / weekly["Close"].iloc[-14] - 1)
    # Strict weekly gate: negative 20W and 50W VFI averages confirm a sustained
    # negative regime; current VFI must be within +/-2 points of zero; the last
    # four completed weeks must rise monotonically with a linear fit R² >= 0.65.
    # A small positive current VFI is allowed if both averages remain negative,
    # capturing a recent zero-cross without admitting an extended move.
    qualifies = bool(
        np.isfinite(value) and abs(value) <= 2.0
        and np.isfinite(vfi_ma50) and vfi_ma50 < 0
        and np.isfinite(vfi_ma20) and vfi_ma20 < 0
        and np.isfinite(vfi.get("VFI_Slope_4B", np.nan)) and vfi["VFI_Slope_4B"] > 0
        and np.isfinite(slope8) and slope8 > 0
        and monotonic4 and np.isfinite(r2_4) and r2_4 >= 0.65
        and np.isfinite(price_return_13w) and price_return_13w <= 0.10
    )
    # Rank qualifying names by closeness to zero, then by longer/shorter slope.
    score = float(100.0 / (1.0 + abs(value))) if np.isfinite(value) else 0.0
    return {
        "Symbol": symbol,
        "Weekly_Screen_Status": "CANDIDATE" if qualifies else "NOT_SELECTED",
        "Weekly_Score": score,
        "Weekly_VFI": value,
        "Weekly_VFI_MA50": vfi_ma50,
        "Weekly_VFI_MA20": vfi_ma20,
        "Weekly_VFI_Prev": vfi.get("VFI_Prev"),
        "Weekly_VFI_Signal": vfi.get("VFI_Signal"),
        "Weekly_VFI_Histogram": vfi.get("VFI_Histogram"),
        "Weekly_VFI_Last4_Monotonic_Rise": monotonic4,
        "Weekly_VFI_R2_4W": r2_4,
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
        ["VFI_Distance_To_Zero", "Weekly_VFI_R2_4W", "Weekly_VFI_Slope_8W", "Weekly_VFI_Slope_20W"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    candidates.insert(0, "Rank", range(1, len(candidates) + 1))
    candidates.to_csv(out / "us_weekly_vfi_candidates.csv", index=False)
    print("\nWEEKLY VFI CANDIDATES: VFI < 0, rising 8W/20W trend, straight last 4 weeks, price not extended", flush=True)
    print(candidates.to_string(index=False) if not candidates.empty else "No assets passed the exact weekly VFI filter.", flush=True)

    # Stage 2: only weekly candidates are examined on completed daily candles.
    # Daily data is never used to admit an asset into the weekly shortlist.
    daily_rows = []
    followup_symbols = candidates["Symbol"].head(max(0, args.max_daily_candidates)).tolist()
    print(f"\\nDAILY FOLLOW-UP for {len(followup_symbols)} weekly candidates", flush=True)
    for i, symbol in enumerate(followup_symbols, 1):
        print(f"[DAILY {i}/{len(followup_symbols)}] {symbol}", flush=True)
        try:
            daily = completed_daily(raw_data[symbol])
            vfi = vfi_snapshot(daily)
            if "VFI" not in vfi:
                raise ValueError(f"Unable to calculate daily VFI: {vfi.get('VFI_Error', 'insufficient data')}")
            flow = calculate_indicators(daily, window=20)
            close = pd.to_numeric(daily["Close"], errors="coerce").dropna()
            ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else float("nan")
            divergence = VolumeDivergenceMM().calculate(daily)
            recent_div = divergence.tail(20)
            bull_div = bool(recent_div["Volume_Div_Bullish"].fillna(False).any())
            last_bull_age = pd.to_numeric(divergence["Volume_Div_Bullish_Age_Bars"], errors="coerce").iloc[-1]
            daily_vfi = float(vfi["VFI"])
            slope4 = float(vfi.get("VFI_Slope_4B", np.nan))
            slope8 = float(vfi.get("VFI_Slope_8B", np.nan))
            # Daily early-accumulation filter: flow improving below zero,
            # accumulation indicators not bearish, and price not already extended.
            early = bool(
                abs(daily_vfi) <= 2.0 and np.isfinite(slope4) and slope4 > 0
                and np.isfinite(slope8) and slope8 > 0
                and np.isfinite(flow["Accumulation_Distribution_Score"])
                and flow["Accumulation_Distribution_Score"] >= 50
                and np.isfinite(ret20) and ret20 <= 0.10
            )
            vfi_proximity = 100.0 / (1.0 + abs(daily_vfi))
            slope_points = 100.0 if slope4 > 0 and slope8 > 0 else (50.0 if slope4 > 0 or slope8 > 0 else 0.0)
            divergence_points = 100.0 if bull_div else 0.0
            daily_score = (
                0.40 * float(flow["Accumulation_Distribution_Score"])
                + 0.20 * vfi_proximity
                + 0.20 * slope_points
                + 0.20 * divergence_points
            )
            daily_rows.append({
                "Symbol": symbol,
                "Weekly_Rank": int(candidates.loc[candidates["Symbol"].eq(symbol), "Rank"].iloc[0]),
                "Daily_Early_Accumulation_Candidate": early,
                "Daily_Score": daily_score,
                "Daily_Candle_Date": daily.index[-1].strftime("%Y-%m-%d"),
                "Daily_Price": float(close.iloc[-1]),
                "Daily_Price_Return_20B": ret20,
                "Daily_VFI": daily_vfi,
                "Daily_VFI_Slope_4D": slope4,
                "Daily_VFI_Slope_8D": slope8,
                "Daily_VFI_Slope_20D": vfi.get("VFI_Slope_20B"),
                "Daily_VFI_Zero_Cross_Recent_4D": vfi.get("VFI_Zero_Cross_Recent_4B"),
                "Daily_Bullish_Volume_Divergence_20D": bull_div,
                "Daily_Last_Bullish_Divergence_Age_D": float(last_bull_age) if np.isfinite(last_bull_age) else np.nan,
                "Daily_Accumulation_Distribution_Score": flow["Accumulation_Distribution_Score"],
                "Daily_OBV_Trend_Score": flow["OBV_Trend_Score"],
                "Daily_AD_Trend_Score": flow["AD_Trend_Score"],
                "Daily_CMF_20": flow["CMF_20"],
                "Daily_MFI_14": flow["MFI_14"],
                "Daily_Relative_Volume_20D": flow["Relative_Volume_20D"],
            })
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "DAILY_FOLLOWUP", "Error": repr(exc)})
            daily_rows.append({"Symbol": symbol, "Weekly_Rank": int(candidates.loc[candidates["Symbol"].eq(symbol), "Rank"].iloc[0]),
                               "Daily_Early_Accumulation_Candidate": False, "Daily_Status": f"ERROR: {exc!r}"})

    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df = daily_df.sort_values(
            ["Daily_Early_Accumulation_Candidate", "Daily_Score", "Weekly_Rank"],
            ascending=[False, False, True], na_position="last"
        ).reset_index(drop=True)
        daily_df.insert(0, "Daily_Rank", range(1, len(daily_df) + 1))
    daily_df.to_csv(out / "us_daily_followup_ranking.csv", index=False)
    daily_df[daily_df.get("Daily_Early_Accumulation_Candidate", pd.Series(index=daily_df.index, dtype=bool)).eq(True)].to_csv(
        out / "us_daily_early_accumulation_candidates.csv", index=False
    )
    print("\\nDAILY FOLLOW-UP RANKING (only weekly candidates)", flush=True)
    print(daily_df.to_string(index=False) if not daily_df.empty else "No weekly candidates to follow up daily.", flush=True)

    summary = {
        "seed_universe_size": len(symbols), "requested_top_liquid": args.top,
        "selected_by_20_session_dollar_turnover": len(selected_symbols),
        "weekly_analyzed": int(weekly_df["Weekly_Screen_Status"].ne("ERROR").sum()) if not weekly_df.empty else 0,
        "weekly_vfi_candidates": int(len(candidates)),
        "daily_followup_analyzed": int(len(daily_df)),
        "daily_early_accumulation_candidates": int(daily_df.get("Daily_Early_Accumulation_Candidate", pd.Series(dtype=bool)).eq(True).sum()),
        "weekly_rule": "completed weekly candles; VFI within +/-2 of zero; VFI 20W and 50W averages both negative; positive 4W and 8W slopes; last 4 weekly VFI points strictly increasing with linear-fit R2 >= 0.65; 13W price return <= +10%",
        "daily_rule": "only strict weekly candidates are followed up on completed daily candles; daily VFI within +/-2 of zero with positive 4D/8D slopes, accumulation/distribution composite >=50 and 20-session price return <= +10%; bullish divergence contributes to ranking; score is a heuristic, not a probability",
        "weekly_candle_policy": "completed weekly candles only; incomplete current week excluded",
        "universe_note": "Top N by mean Close*Volume over 20 completed sessions within a curated US-listed seed universe; not all US listings.",
        "method_note": "VFI-only selection. OHLCV-based volume indicator; not direct institutional-flow data.",
        "errors": errors,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nRUN SUMMARY\n" + json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
