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
    if len(s) < bars: return float("nan")
    return float(np.polyfit(np.arange(len(s)), s.to_numpy(dtype=float), 1)[0])

def vfi_linearity_metrics(series, bars=13):
    """Score an approximately linear, low-noise upward VFI path over recent bars."""
    values = pd.to_numeric(series, errors="coerce").dropna().tail(bars).to_numpy(dtype=float)
    if len(values) < max(8, bars // 2):
        return {"VFI_Linear_R2": float("nan"), "VFI_Up_Bar_Ratio": float("nan"),
                "VFI_Smooth_Rising_Score": float("nan"), "VFI_Linear_Slope": float("nan")}
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = intercept + slope * x
    ss_res = float(np.sum((values - fitted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    diffs = np.diff(values)
    up_ratio = float(np.mean(diffs > 0)) if len(diffs) else 0.0
    # R² alone would reward a straight decline too, so require a positive slope.
    score = float(100.0 * r2 * up_ratio) if slope > 0 else 0.0
    return {"VFI_Linear_R2": float(r2), "VFI_Up_Bar_Ratio": up_ratio,
            "VFI_Smooth_Rising_Score": score, "VFI_Linear_Slope": float(slope)}


def vfi_snapshot(data, smooth_window=13):
    try:
        v = VFIIndicator().calculate(data)
        valid = v.loc[v["VFI"].notna()]
        if len(valid) < 10: return {}
        s = valid["VFI"]
        latest = valid.iloc[-1]
        last_vfi = float(s.iloc[-1])
        prior_vfi = float(s.iloc[-2]) if len(s) >= 2 else float("nan")
        tail26 = s.tail(26)
        min26 = float(tail26.min()) if len(tail26) else float("nan")
        recent4 = s.tail(4)
        crossed_zero_4b = bool(
            len(recent4) >= 2 and
            ((recent4.shift(1) < 0) & (recent4 >= 0)).any()
        )
        smooth = vfi_linearity_metrics(s, bars=smooth_window)
        return {
            "VFI": last_vfi,
            **smooth,
            "VFI_Prev": prior_vfi,
            "VFI_Signal": float(latest.get("VFI_Signal", np.nan)),
            "VFI_Histogram": float(latest.get("VFI_Histogram", np.nan)),
            "VFI_Slope_4B": safe_slope(s, 4),
            "VFI_Slope_8B": safe_slope(s, 8),
            "VFI_Improving": bool(safe_slope(s, 4) > 0 and safe_slope(s, 8) > 0),
            "VFI_Near_Zero": bool(abs(last_vfi) <= 2.0),
            "VFI_Min_26B": min26,
            "VFI_Rebound_From_Min_26B": last_vfi - min26 if np.isfinite(min26) else float("nan"),
            "VFI_Bars_Below_Zero_26B": int((tail26 < 0).sum()),
            "VFI_Zero_Cross_Recent_4B": crossed_zero_4b,
            "VFI_State": str(latest.get("VFI_State", "")),
        }
    except Exception as exc:
        return {"VFI_Error": repr(exc)}

def weekly_screen(symbol, daily):
    daily = completed_daily(daily)
    weekly = to_weekly(daily)
    if len(weekly) < 40:
        raise ValueError(f"Only {len(weekly)} completed weekly bars; need 40")
    flow = calculate_indicators(weekly, window=13)
    # Require a genuinely extended weekly trajectory: 20 completed weekly candles (~5 months),
    # rather than allowing a short 13-week rebound to dominate the ranking.
    vfi = vfi_snapshot(weekly, smooth_window=20)
    div = {}
    try:
        d = VolumeDivergenceMM().calculate(weekly)
        valid = d.loc[d["Volume_Divergence_MM"].notna()]
        if not valid.empty:
            s = valid["Volume_Divergence_MM"]
            div = {"Weekly_Divergence": float(s.iloc[-1]),
                   "Weekly_Divergence_Slope_4B": safe_slope(s, 4),
                   "Weekly_Divergence_Slope_8B": safe_slope(s, 8)}
    except Exception as exc:
        div = {"Weekly_Divergence_Error": repr(exc)}
    price_ret_13w = float(weekly["Close"].iloc[-1] / weekly["Close"].iloc[-14] - 1)
    flow_score = flow["Accumulation_Distribution_Score"]
    flow_trend_score = np.nanmean([flow.get("OBV_Trend_Score", np.nan), flow.get("AD_Trend_Score", np.nan)])
    # Early-entry ranking rewards VFI below but approaching zero, improving
    # short/medium slopes, and a price that has not already run up.
    vfi_value = vfi.get("VFI", np.nan)
    vfi_slope4 = vfi.get("VFI_Slope_4B", np.nan)
    vfi_slope8 = vfi.get("VFI_Slope_8B", np.nan)
    if np.isfinite(vfi_value):
        vfi_position_score = (max(0.0, 100.0 - abs(vfi_value) * 4.0) if vfi_value <= 0
                              else max(0.0, 50.0 - vfi_value * 2.5))
    else:
        vfi_position_score = 50.0
    if np.isfinite(vfi_slope4) and np.isfinite(vfi_slope8):
        vfi_slope_score = 100.0 if vfi_slope4 > 0 and vfi_slope8 > 0 else (
            60.0 if vfi_slope4 > 0 or vfi_slope8 > 0 else 0.0)
    else:
        vfi_slope_score = 50.0
    if price_ret_13w <= 0:
        early_price_score = 90.0
    elif price_ret_13w <= 0.05:
        early_price_score = 75.0
    elif price_ret_13w <= 0.15:
        early_price_score = 55.0
    else:
        early_price_score = max(0.0, 55.0 - (price_ret_13w - 0.15) * 200.0)
    smooth_score = vfi.get("VFI_Smooth_Rising_Score", np.nan)
    if not np.isfinite(smooth_score):
        smooth_score = 0.0
    weekly_rank_score = float(
        0.25 * flow_score + 0.10 * flow_trend_score
        + 0.25 * vfi_position_score + 0.15 * vfi_slope_score
        + 0.20 * smooth_score + 0.05 * early_price_score
    )
    # Strict weekly gate: VFI must still be below zero, close enough to be
    # approaching it, and accelerating over both short and medium windows.
    qualifies = bool(
        np.isfinite(weekly_rank_score) and weekly_rank_score >= 48
        and np.isfinite(vfi_value) and -15.0 <= vfi_value <= 0.0
        and np.isfinite(vfi_slope4) and vfi_slope4 > 0
        and np.isfinite(vfi_slope8) and vfi_slope8 > 0
        and flow_score >= 35
        and price_ret_13w <= 0.15
        and smooth_score >= 45
    )
    # A separate early-turn watchlist catches a weekly turn that has begun in
    # the latest 4/8 bars but is not yet smooth over the full 13-bar window.
    # These are investigated daily, but are not labelled strict candidates.
    turn_watch = bool(
        not qualifies
        and np.isfinite(weekly_rank_score) and weekly_rank_score >= 48
        and np.isfinite(vfi_value) and -15.0 <= vfi_value <= 0.0
        and np.isfinite(vfi_slope4) and vfi_slope4 > 0
        and np.isfinite(vfi_slope8) and vfi_slope8 > 0
        and flow_score >= 35
        and price_ret_13w <= 0.15
    )
    return {
        "Symbol": symbol, "Weekly_Score": float(weekly_rank_score),
        "Weekly_VFI_Smooth_Rising_Score": smooth_score,
        "Weekly_VFI_Linear_R2": vfi.get("VFI_Linear_R2"),
        "Weekly_VFI_Up_Bar_Ratio": vfi.get("VFI_Up_Bar_Ratio"),
        "Weekly_VFI_Linear_Slope": vfi.get("VFI_Linear_Slope"),
        "Weekly_Early_Accumulation_Score": float(weekly_rank_score),
        "Weekly_VFI_Position_Score": float(vfi_position_score),
        "Weekly_VFI_Slope_Score": float(vfi_slope_score),
        "Weekly_Flow_Score": float(flow_score),
        "Weekly_Price_Return_13W": price_ret_13w,
        "Weekly_Candle_Date": weekly.index[-1].strftime("%Y-%m-%d"),
        "Weekly_OBV_Score": flow.get("OBV_Trend_Score"),
        "Weekly_AD_Score": flow.get("AD_Trend_Score"),
        "Weekly_CMF": flow.get("CMF_20"), "Weekly_MFI": flow.get("MFI_14"),
        **{f"Weekly_{k}": val for k, val in vfi.items()},
        **div, "_qualifies": qualifies, "_turn_watch": turn_watch,
        "_weekly": weekly, "_daily": daily,
    }

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=",".join(US_LIQUID_UNIVERSE))
    p.add_argument("--period", default="max")
    p.add_argument("--top", type=int, default=100, help="Analyze the top N names ranked by recent dollar turnover within the curated seed universe.")
    p.add_argument("--max-daily-candidates", type=int, default=30)
    p.add_argument("--output-dir", default="reports/us_weekly_then_daily")
    args = p.parse_args()
    symbols = list(dict.fromkeys(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    seed_symbols = symbols
    service = MarketDataService()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_data, liquidity_rows, errors = {}, [], []
    print(f"Downloading data for {len(seed_symbols)} liquid US-listed seed symbols to rank turnover...", flush=True)
    for i, symbol in enumerate(seed_symbols, 1):
        print(f"[LIQUIDITY {i}/{len(seed_symbols)}] {symbol}", flush=True)
        try:
            raw = service.get_daily_data(symbol, period=args.period)
            daily = completed_daily(raw)
            raw_data[symbol] = daily
            dollar_turnover = (pd.to_numeric(daily["Close"], errors="coerce") *
                               pd.to_numeric(daily["Volume"], errors="coerce")).replace([np.inf, -np.inf], np.nan).dropna().tail(20)
            turnover = float(dollar_turnover.mean()) if len(dollar_turnover) >= 10 else float("nan")
            liquidity_rows.append({"Symbol": symbol, "Average_Dollar_Turnover_20D": turnover,
                                   "Liquidity_Data_Date": daily.index[-1].strftime("%Y-%m-%d") if not daily.empty else "",
                                   "Liquidity_Status": "OK" if np.isfinite(turnover) else "INSUFFICIENT_DATA"})
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "LIQUIDITY", "Error": repr(exc)})
            liquidity_rows.append({"Symbol": symbol, "Average_Dollar_Turnover_20D": np.nan,
                                   "Liquidity_Data_Date": "", "Liquidity_Status": f"ERROR: {exc!r}"})
    liquidity_df = pd.DataFrame(liquidity_rows)
    eligible = liquidity_df[np.isfinite(pd.to_numeric(liquidity_df["Average_Dollar_Turnover_20D"], errors="coerce"))]
    selected = eligible.sort_values("Average_Dollar_Turnover_20D", ascending=False).head(args.top).copy()
    selected_symbols = selected["Symbol"].tolist()
    liquidity_rank = {row.Symbol: (rank, float(row.Average_Dollar_Turnover_20D))
                      for rank, row in enumerate(selected.itertuples(index=False), 1)}
    print(f"Liquidity universe={len(seed_symbols)}; selected top {len(selected_symbols)} by 20-session dollar turnover.", flush=True)

    weekly_rows, candidate_frames = [], {}
    for i, symbol in enumerate(selected_symbols, 1):
        print(f"[WEEKLY {i}/{len(selected_symbols)}] {symbol}", flush=True)
        try:
            row = weekly_screen(symbol, raw_data[symbol])
            candidate_frames[symbol] = (row.pop("_daily"), row.pop("_weekly"))
            qualifies = row.pop("_qualifies")
            turn_watch = row.pop("_turn_watch")
            row["Weekly_Screen_Status"] = (
                "CANDIDATE" if qualifies else
                "EARLY_TURN_WATCH" if turn_watch else
                "NOT_SELECTED"
            )
            row["Liquidity_Rank"] = liquidity_rank[symbol][0]
            row["Average_Dollar_Turnover_20D"] = liquidity_rank[symbol][1]
            weekly_rows.append(row)
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "WEEKLY", "Error": repr(exc)})
            weekly_rows.append({"Symbol": symbol, "Weekly_Screen_Status": "ERROR", "Error": repr(exc),
                                "Liquidity_Rank": liquidity_rank[symbol][0],
                                "Average_Dollar_Turnover_20D": liquidity_rank[symbol][1]})
    weekly_df = pd.DataFrame(weekly_rows)
    rankable = weekly_df[weekly_df["Weekly_Screen_Status"].eq("CANDIDATE")].copy() if "Weekly_Screen_Status" in weekly_df else pd.DataFrame()
    if not rankable.empty:
        rankable = rankable.sort_values("Weekly_Score", ascending=False)
    # Daily analysis follows the weekly screen: strict candidates first, then
    # the near-zero watchlist, then early turns whose latest 4/8 weekly slopes
    # are positive but whose full 13-week VFI path is not yet smooth.
    strict_symbols = rankable.sort_values("Weekly_Score", ascending=False)["Symbol"].tolist() if not rankable.empty else []
    valid_weekly = weekly_df[weekly_df["Weekly_Screen_Status"].ne("ERROR")].copy()
    vfi_num = pd.to_numeric(valid_weekly.get("Weekly_VFI", pd.Series(np.nan, index=valid_weekly.index)), errors="coerce")
    slope4_num = pd.to_numeric(valid_weekly.get("Weekly_VFI_Slope_4B", pd.Series(np.nan, index=valid_weekly.index)), errors="coerce")
    ret13_num = pd.to_numeric(valid_weekly.get("Weekly_Price_Return_13W", pd.Series(np.nan, index=valid_weekly.index)), errors="coerce")
    smooth_num = pd.to_numeric(valid_weekly.get("Weekly_VFI_Smooth_Rising_Score", pd.Series(np.nan, index=valid_weekly.index)), errors="coerce")
    slope8_num = pd.to_numeric(valid_weekly.get("Weekly_VFI_Slope_8B", pd.Series(np.nan, index=valid_weekly.index)), errors="coerce")
    near_zero_mask = (vfi_num.between(-15.0, 0.0) & (slope4_num > 0) &
                      (slope8_num > 0) & (ret13_num <= 0.15) & (smooth_num >= 35))
    near_zero = valid_weekly.loc[near_zero_mask].copy()
    near_zero["_VFI_Abs"] = pd.to_numeric(near_zero["Weekly_VFI"], errors="coerce").abs()
    near_zero = near_zero.sort_values(["_VFI_Abs", "Weekly_Score"], ascending=[True, False])
    watchlist_symbols = near_zero["Symbol"].tolist()

    # Also investigate a *recent* weekly zero-cross, but only while price has
    # not already run up. This captures an early turn just over zero without
    # admitting extended rallies such as names up >15% over 13 weeks.
    cross_flag = valid_weekly.get(
        "Weekly_VFI_Zero_Cross_Recent_4B", pd.Series(False, index=valid_weekly.index)
    ).fillna(False).astype(bool)
    below_zero_bars = pd.to_numeric(
        valid_weekly.get("Weekly_VFI_Bars_Below_Zero_26B", pd.Series(0, index=valid_weekly.index)),
        errors="coerce",
    )
    recent_cross_mask = (
        vfi_num.between(0.0, 3.0) & cross_flag & (slope4_num > 0) & (slope8_num > 0)
        & (ret13_num <= 0.15) & (smooth_num >= 35) & (below_zero_bars >= 8)
    )
    recent_cross = valid_weekly.loc[recent_cross_mask].copy()
    if not recent_cross.empty:
        recent_cross["_VFI_Abs"] = pd.to_numeric(recent_cross["Weekly_VFI"], errors="coerce").abs()
        recent_cross = recent_cross.sort_values(["_VFI_Abs", "Weekly_Score"], ascending=[True, False])
    recent_cross_symbols = recent_cross["Symbol"].tolist()
    for crossed_symbol in recent_cross_symbols:
        weekly_df.loc[weekly_df["Symbol"].eq(crossed_symbol), "Weekly_Screen_Status"] = "RECENT_ZERO_CROSS_WATCH"

    turn_watch_symbols = weekly_df.loc[
        weekly_df["Weekly_Screen_Status"].eq("EARLY_TURN_WATCH")
    ].sort_values("Weekly_Score", ascending=False)["Symbol"].tolist()
    chosen = list(dict.fromkeys(
        strict_symbols + watchlist_symbols + recent_cross_symbols + turn_watch_symbols
    ))[:args.max_daily_candidates]
    if strict_symbols and (watchlist_symbols or recent_cross_symbols or turn_watch_symbols):
        daily_selection_mode = "STRICT_PLUS_WEEKLY_WATCHLIST"
    elif strict_symbols:
        daily_selection_mode = "STRICT_WEEKLY_CANDIDATES"
    elif watchlist_symbols or recent_cross_symbols or turn_watch_symbols:
        daily_selection_mode = "WEEKLY_EARLY_ACCUMULATION_WATCHLIST"
    else:
        # Do not silently analyse arbitrary high-score assets when none passed
        # the weekly gate; that would break the requested weekly-first process.
        daily_selection_mode = "NO_WEEKLY_CANDIDATES"
        chosen = []
    print(f"Weekly universe={len(selected_symbols)}; strict weekly candidates={len(rankable)}; early-turn watchlist={len(turn_watch_symbols)}; daily deep-dive={len(chosen)}; mode={daily_selection_mode}", flush=True)
    print("TOP WEEKLY EARLY-ACCUMULATION RANKING (not buy signals):", flush=True)
    weekly_cols = [c for c in ["Liquidity_Rank", "Symbol", "Average_Dollar_Turnover_20D", "Weekly_Score", "Weekly_Early_Accumulation_Score", "Weekly_Flow_Score", "Weekly_Price_Return_13W", "Weekly_VFI", "Weekly_VFI_Prev", "Weekly_VFI_Signal", "Weekly_VFI_Histogram", "Weekly_VFI_Slope_4B", "Weekly_VFI_Slope_8B", "Weekly_VFI_Smooth_Rising_Score", "Weekly_VFI_Linear_R2", "Weekly_VFI_Up_Bar_Ratio", "Weekly_VFI_Position_Score", "Weekly_VFI_Slope_Score", "Weekly_VFI_Near_Zero", "Weekly_VFI_Min_26B", "Weekly_VFI_Rebound_From_Min_26B", "Weekly_VFI_Bars_Below_Zero_26B", "Weekly_VFI_Zero_Cross_Recent_4B", "Weekly_Candle_Date", "Weekly_Screen_Status"] if c in weekly_df.columns]
    if not weekly_df.empty and "Weekly_Score" in weekly_df.columns:
        print(weekly_df.sort_values("Weekly_Score", ascending=False, na_position="last")[weekly_cols].head(20).to_string(index=False), flush=True)
    daily_rows = []
    for i, symbol in enumerate(chosen, 1):
        print(f"[DAILY {i}/{len(chosen)}] {symbol}", flush=True)
        daily, weekly = candidate_frames[symbol]
        try:
            dflow = calculate_indicators(daily, window=20)
            dvfi = vfi_snapshot(daily, smooth_window=20)
            price_ret_20 = dflow.get("Price_Return_20B", np.nan)
            dscore = float(dflow["Accumulation_Distribution_Score"])
            vfi = dvfi.get("VFI", np.nan)
            vfi_slope4 = dvfi.get("VFI_Slope_4B", np.nan)
            vfi_slope8 = dvfi.get("VFI_Slope_8B", np.nan)
            # Strict daily gate: do not promote a signal if VFI has already
            # crossed zero or if only the shortest slope is positive.
            smooth_daily = dvfi.get("VFI_Smooth_Rising_Score", np.nan)
            daily_zero_cross = bool(dvfi.get("VFI_Zero_Cross_Recent_4B", False))
            daily_pre_zero = bool(np.isfinite(vfi) and -15.0 <= vfi <= 0.0)
            daily_post_zero = bool(np.isfinite(vfi) and 0.0 < vfi <= 2.0 and daily_zero_cross)
            vfi_early = bool(
                (daily_pre_zero or daily_post_zero)
                and np.isfinite(vfi_slope4) and vfi_slope4 > 0
                and np.isfinite(vfi_slope8) and vfi_slope8 > 0
                and price_ret_20 <= 0.10
                and dscore >= 45
                and np.isfinite(smooth_daily) and smooth_daily >= 55
            )
            near_zero_watch = bool(
                (np.isfinite(vfi) and -5.0 <= vfi <= 0.0 or daily_post_zero)
                and np.isfinite(vfi_slope4) and vfi_slope4 > 0
                and np.isfinite(vfi_slope8) and vfi_slope8 > 0
                and price_ret_20 <= 0.10
                and np.isfinite(smooth_daily) and smooth_daily >= 40
            )
            early_vfi_score = 100 if vfi_early else (75 if near_zero_watch else 20)
            extension_penalty = 20 if price_ret_20 > 0.20 else (10 if price_ret_20 > 0.12 else 0)
            final_score = max(0, min(100, 0.45 * float(weekly_df.loc[weekly_df["Symbol"].eq(symbol), "Weekly_Score"].iloc[0])
                                     + 0.35 * dscore + 0.20 * early_vfi_score - extension_penalty))
            daily_rows.append({
                "Symbol": symbol, "VFI_Early_Accumulation_Qualified": vfi_early,
                "Final_Rank_Score": final_score,
                "Weekly_Score": float(weekly_df.loc[weekly_df["Symbol"].eq(symbol), "Weekly_Score"].iloc[0]),
                "Daily_Flow_Score": dscore, "Daily_Price": float(daily["Close"].iloc[-1]),
                "Daily_Candle_Date": daily.index[-1].strftime("%Y-%m-%d"),
                "Daily_Price_Return_20B": price_ret_20,
                "Daily_OBV_Score": dflow.get("OBV_Trend_Score"), "Daily_AD_Score": dflow.get("AD_Trend_Score"),
                "Daily_CMF": dflow.get("CMF_20"), "Daily_MFI": dflow.get("MFI_14"),
                "Daily_VFI": vfi, "Daily_VFI_Prev": dvfi.get("VFI_Prev"),
                "Daily_VFI_Signal": dvfi.get("VFI_Signal"), "Daily_VFI_Histogram": dvfi.get("VFI_Histogram"),
                "Daily_VFI_Slope_4B": vfi_slope4, "Daily_VFI_Slope_8B": vfi_slope8,
                "Daily_VFI_Smooth_Rising_Score": smooth_daily,
                "Daily_VFI_Linear_R2": dvfi.get("VFI_Linear_R2"),
                "Daily_VFI_Up_Bar_Ratio": dvfi.get("VFI_Up_Bar_Ratio"),
                "Daily_VFI_Linear_Slope": dvfi.get("VFI_Linear_Slope"),
                "Daily_VFI_Min_26B": dvfi.get("VFI_Min_26B"),
                "Daily_VFI_Rebound_From_Min_26B": dvfi.get("VFI_Rebound_From_Min_26B"),
                "Daily_VFI_Bars_Below_Zero_26B": dvfi.get("VFI_Bars_Below_Zero_26B"),
                "Daily_VFI_Zero_Cross_Recent_4B": dvfi.get("VFI_Zero_Cross_Recent_4B"),
                "Daily_VFI_Near_Zero": dvfi.get("VFI_Near_Zero"),
                "Weekly_Price_Return_13W": float(weekly_df.loc[weekly_df["Symbol"].eq(symbol), "Weekly_Price_Return_13W"].iloc[0]),
                "Weekly_Candle_Date": weekly_df.loc[weekly_df["Symbol"].eq(symbol), "Weekly_Candle_Date"].iloc[0],
                "Setup_Interpretation": (
                    "EARLY_ACCUMULATION_CANDIDATE_RECENT_ZERO_CROSS" if vfi_early and daily_post_zero else
                    "EARLY_ACCUMULATION_CANDIDATE_PRE_ZERO" if vfi_early else
                    "RECENT_ZERO_CROSS_WATCH" if near_zero_watch and daily_post_zero else
                    "SMOOTH_NEAR_ZERO_WATCH" if near_zero_watch else
                    "DOES_NOT_MEET_STRICT_VFI_FILTER"
                ),
            })
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "DAILY", "Error": repr(exc)})
    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df = daily_df.sort_values(
            ["VFI_Early_Accumulation_Qualified", "Final_Rank_Score"],
            ascending=[False, False]
        ).reset_index(drop=True)
        daily_df.insert(0, "Rank", range(1, len(daily_df) + 1))
    strict_daily = daily_df[daily_df["VFI_Early_Accumulation_Qualified"].eq(True)].copy() if not daily_df.empty else daily_df.copy()
    # Explicit weekly-first ranking: require the weekly VFI early-turn gate,
    # then confirm the Volume Divergence MM line is rising on both 4- and 8-week
    # windows. Rank qualifying names by absolute VFI (closest to zero first).
    vfi_divergence_ranking = weekly_df.copy()
    for col in ["Weekly_VFI", "Weekly_VFI_Slope_4B", "Weekly_VFI_Slope_8B",
                "Weekly_Price_Return_13W", "Weekly_Divergence_Slope_4B",
                "Weekly_Divergence_Slope_8B", "Weekly_Score"]:
        if col in vfi_divergence_ranking.columns:
            vfi_divergence_ranking[col] = pd.to_numeric(vfi_divergence_ranking[col], errors="coerce")
    vfi_divergence_ranking = vfi_divergence_ranking[
        vfi_divergence_ranking["Weekly_Screen_Status"].isin(
            ["CANDIDATE", "EARLY_TURN_WATCH", "RECENT_ZERO_CROSS_WATCH"]
        )
        & vfi_divergence_ranking["Weekly_VFI"].between(-15.0, 0.0)
        & (vfi_divergence_ranking["Weekly_VFI_Slope_4B"] > 0)
        & (vfi_divergence_ranking["Weekly_VFI_Slope_8B"] > 0)
        & (vfi_divergence_ranking["Weekly_Price_Return_13W"] <= 0.15)
        & (vfi_divergence_ranking["Weekly_Divergence_Slope_4B"] > 0)
        & (vfi_divergence_ranking["Weekly_Divergence_Slope_8B"] > 0)
    ].copy()
    vfi_divergence_ranking["VFI_Distance_To_Zero"] = vfi_divergence_ranking["Weekly_VFI"].abs()
    vfi_divergence_ranking = vfi_divergence_ranking.sort_values(
        ["VFI_Distance_To_Zero", "Weekly_Score"], ascending=[True, False]
    )
    vfi_divergence_cols = [
        "Symbol", "Weekly_Screen_Status", "Liquidity_Rank", "Average_Dollar_Turnover_20D",
        "Weekly_Score", "Weekly_Candle_Date", "Weekly_VFI", "Weekly_VFI_Prev",
        "Weekly_VFI_Signal", "Weekly_VFI_Histogram", "Weekly_VFI_Slope_4B",
        "Weekly_VFI_Slope_8B", "Weekly_VFI_Smooth_Rising_Score",
        "Weekly_Price_Return_13W", "Weekly_Divergence", "Weekly_Divergence_Slope_4B",
        "Weekly_Divergence_Slope_8B", "VFI_Distance_To_Zero"
    ]
    vfi_divergence_ranking = vfi_divergence_ranking[
        [c for c in vfi_divergence_cols if c in vfi_divergence_ranking.columns]
    ].reset_index(drop=True)
    vfi_divergence_ranking.insert(0, "Rank", range(1, len(vfi_divergence_ranking) + 1))
    vfi_divergence_ranking.to_csv(out / "us_weekly_vfi_divergence_ranking.csv", index=False)
    print("WEEKLY VFI + DIVERGENCE CONFIRMATION RANKING:", flush=True)
    print(vfi_divergence_ranking.to_string(index=False) if not vfi_divergence_ranking.empty
          else "No assets passed both the weekly VFI early-turn filter and positive 4/8-week divergence slopes.", flush=True)

    weekly_out = weekly_df.drop(columns=[], errors="ignore")
    weekly_out.to_csv(out / "us_100_weekly_screen.csv", index=False)
    liquidity_df.sort_values("Average_Dollar_Turnover_20D", ascending=False).to_csv(out / "us_liquidity_universe.csv", index=False)
    daily_df.to_csv(out / "us_daily_shortlist_ranking.csv", index=False)
    strict_daily.to_csv(out / "us_daily_strict_vfi_candidates.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps({
        "seed_universe_size": len(seed_symbols), "requested_top_liquid": args.top,
        "selected_by_20_session_dollar_turnover": len(selected_symbols),
        "weekly_analyzed": int(weekly_df["Weekly_Screen_Status"].ne("ERROR").sum()) if "Weekly_Screen_Status" in weekly_df else 0,
        "weekly_candidates": len(rankable),
        "weekly_early_turn_watchlist": int(weekly_df["Weekly_Screen_Status"].eq("EARLY_TURN_WATCH").sum()) if "Weekly_Screen_Status" in weekly_df else 0,
        "weekly_recent_zero_cross_watchlist": int(weekly_df["Weekly_Screen_Status"].eq("RECENT_ZERO_CROSS_WATCH").sum()) if "Weekly_Screen_Status" in weekly_df else 0,
        "daily_analyzed": len(daily_df), "daily_selection_mode": daily_selection_mode,
        "strict_daily_vfi_candidates": int(len(strict_daily)),
        "weekly_candle_policy": "completed Friday candles only",
        "daily_candle_policy": "completed daily candles only",
        "universe_note": "Top N by mean Close*Volume over 20 completed sessions within the curated seed universe; not all US listings.",
        "liquidity_ranking_csv": "us_liquidity_universe.csv",
        "method_note": "Technical OHLCV proxies; not direct institutional-flow data. Final score is heuristic and requires out-of-sample backtesting.",
        "errors": errors,
    }, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"seed_universe": len(seed_symbols), "selected_by_liquidity": len(selected_symbols), "weekly_candidates": len(rankable),
                      "weekly_early_turn_watchlist": int(weekly_df["Weekly_Screen_Status"].eq("EARLY_TURN_WATCH").sum()) if "Weekly_Screen_Status" in weekly_df else 0,
                      "weekly_recent_zero_cross_watchlist": int(weekly_df["Weekly_Screen_Status"].eq("RECENT_ZERO_CROSS_WATCH").sum()) if "Weekly_Screen_Status" in weekly_df else 0,
                      "daily_analyzed": len(daily_df), "strict_daily_vfi_candidates": len(strict_daily),
                      "errors": len(errors)}, indent=2))
    if not strict_daily.empty:
        print("STRICT EARLY VFI CANDIDATES:")
        print(strict_daily.to_string(index=False))
    else:
        print("No asset currently meets the strict weekly + daily VFI-from-below criteria.")
        if not daily_df.empty:
            print("Daily analysis of weekly candidates (diagnostic; not buy signals):")
            print(daily_df.to_string(index=False))

if __name__ == "__main__":
    main()
