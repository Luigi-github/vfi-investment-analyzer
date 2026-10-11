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

# Curated universe of highly traded US-listed stocks and ETFs; liquidity is not a live-ranked top-100 feed.
US_LIQUID_UNIVERSE = """
AAPL MSFT NVDA AMZN META GOOGL GOOG TSLA AVGO AMD INTC QCOM MU ORCL CRM PLTR NFLX
JPM BAC WFC C GS MS V MA AXP XOM CVX COP SLB GE CAT BA HON RTX LMT UNH LLY JNJ ABBV MRK PFE AMGN COST WMT TGT HD LOW MCD KO PEP PG
DIS CMCSA T VZ TMUS UBER ABNB SHOP COIN HOOD SQ PYPL SOFI RIVN F FDX UPS NKE SBUX TSM ASML ARM SMCI
SPY QQQ IWM DIA TLT GLD SLV XLF XLK XLE XLV SMH SOXX HYG LQD EEM EFA IBIT
""".split()

def safe_slope(series, bars=4):
    s = pd.to_numeric(series, errors="coerce").dropna().tail(bars)
    if len(s) < bars: return float("nan")
    return float(np.polyfit(np.arange(len(s)), s.to_numpy(dtype=float), 1)[0])

def vfi_snapshot(data):
    try:
        v = VFIIndicator().calculate(data)
        valid = v.loc[v["VFI"].notna()]
        if len(valid) < 10: return {}
        s = valid["VFI"]
        latest = valid.iloc[-1]
        return {
            "VFI": float(s.iloc[-1]),
            "VFI_Slope_4B": safe_slope(s, 4),
            "VFI_Slope_8B": safe_slope(s, 8),
            "VFI_Improving": bool(safe_slope(s, 4) > 0 and safe_slope(s, 8) > 0),
            "VFI_Near_Zero": bool(abs(float(s.iloc[-1])) <= 2.0),
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
    vfi = vfi_snapshot(weekly)
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
    vfi_slope = vfi.get("VFI_Slope_4B", np.nan)
    weekly_rank_score = 0.65 * flow_score + 0.20 * flow_trend_score + 0.15 * (100 if vfi_slope > 0 else 0)
    # Keep early setups: improving flow, price not excessively extended. A negative VFI can still qualify.
    qualifies = bool(
        np.isfinite(weekly_rank_score) and weekly_rank_score >= 48
        and (flow_score >= 48 or vfi_slope > 0 or div.get("Weekly_Divergence_Slope_4B", 0) > 0)
        and price_ret_13w <= 0.25
    )
    return {
        "Symbol": symbol, "Weekly_Score": float(weekly_rank_score),
        "Weekly_Flow_Score": float(flow_score),
        "Weekly_Price_Return_13W": price_ret_13w,
        "Weekly_Candle_Date": weekly.index[-1].strftime("%Y-%m-%d"),
        "Weekly_OBV_Score": flow.get("OBV_Trend_Score"),
        "Weekly_AD_Score": flow.get("AD_Trend_Score"),
        "Weekly_CMF": flow.get("CMF_20"), "Weekly_MFI": flow.get("MFI_14"),
        **{f"Weekly_{k}": val for k, val in vfi.items()},
        **div, "_qualifies": qualifies, "_weekly": weekly,
        "_daily": daily,
    }

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=",".join(US_LIQUID_UNIVERSE))
    p.add_argument("--period", default="max")
    p.add_argument("--max-daily-candidates", type=int, default=30)
    p.add_argument("--output-dir", default="reports/us_weekly_then_daily")
    args = p.parse_args()
    symbols = list(dict.fromkeys(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    service = MarketDataService()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    weekly_rows, candidate_frames, errors = [], {}, []
    for i, symbol in enumerate(symbols, 1):
        print(f"[WEEKLY {i}/{len(symbols)}] {symbol}", flush=True)
        try:
            raw = service.get_daily_data(symbol, period=args.period)
            row = weekly_screen(symbol, raw)
            candidate_frames[symbol] = (row.pop("_daily"), row.pop("_weekly"))
            row["Weekly_Screen_Status"] = "CANDIDATE" if row.pop("_qualifies") else "NOT_SELECTED"
            weekly_rows.append(row)
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "WEEKLY", "Error": repr(exc)})
            weekly_rows.append({"Symbol": symbol, "Weekly_Screen_Status": "ERROR", "Error": repr(exc)})
    weekly_df = pd.DataFrame(weekly_rows)
    rankable = weekly_df[weekly_df["Weekly_Screen_Status"].eq("CANDIDATE")].copy() if "Weekly_Screen_Status" in weekly_df else pd.DataFrame()
    if not rankable.empty:
        rankable = rankable.sort_values("Weekly_Score", ascending=False)
    chosen = rankable["Symbol"].head(args.max_daily_candidates).tolist() if not rankable.empty else []
    print(f"Weekly universe={len(symbols)}; candidates={len(rankable)}; daily deep-dive={len(chosen)}", flush=True)
    daily_rows = []
    for i, symbol in enumerate(chosen, 1):
        print(f"[DAILY {i}/{len(chosen)}] {symbol}", flush=True)
        daily, weekly = candidate_frames[symbol]
        try:
            dflow = calculate_indicators(daily, window=20)
            dvfi = vfi_snapshot(daily)
            price_ret_20 = dflow.get("Price_Return_20B", np.nan)
            dscore = float(dflow["Accumulation_Distribution_Score"])
            vfi = dvfi.get("VFI", np.nan)
            vfi_slope4 = dvfi.get("VFI_Slope_4B", np.nan)
            vfi_slope8 = dvfi.get("VFI_Slope_8B", np.nan)
            # Reward improving negative/near-zero VFI, penalize overextended prices.
            early_vfi_score = 75 if (np.isfinite(vfi) and vfi <= 0 and vfi_slope4 > 0 and vfi_slope8 > 0) else (
                65 if (np.isfinite(vfi) and abs(vfi) <= 2 and vfi_slope4 > 0) else (
                50 if vfi_slope4 > 0 else 20))
            extension_penalty = 20 if price_ret_20 > 0.20 else (10 if price_ret_20 > 0.12 else 0)
            final_score = max(0, min(100, 0.45 * float(weekly_df.loc[weekly_df["Symbol"].eq(symbol), "Weekly_Score"].iloc[0])
                                     + 0.35 * dscore + 0.20 * early_vfi_score - extension_penalty))
            daily_rows.append({
                "Symbol": symbol, "Final_Rank_Score": final_score,
                "Weekly_Score": float(weekly_df.loc[weekly_df["Symbol"].eq(symbol), "Weekly_Score"].iloc[0]),
                "Daily_Flow_Score": dscore, "Daily_Price": float(daily["Close"].iloc[-1]),
                "Daily_Candle_Date": daily.index[-1].strftime("%Y-%m-%d"),
                "Daily_Price_Return_20B": price_ret_20,
                "Daily_OBV_Score": dflow.get("OBV_Trend_Score"), "Daily_AD_Score": dflow.get("AD_Trend_Score"),
                "Daily_CMF": dflow.get("CMF_20"), "Daily_MFI": dflow.get("MFI_14"),
                "Daily_VFI": vfi, "Daily_VFI_Slope_4B": vfi_slope4, "Daily_VFI_Slope_8B": vfi_slope8,
                "Daily_VFI_Near_Zero": dvfi.get("VFI_Near_Zero"),
                "Weekly_Price_Return_13W": float(weekly_df.loc[weekly_df["Symbol"].eq(symbol), "Weekly_Price_Return_13W"].iloc[0]),
                "Weekly_Candle_Date": weekly_df.loc[weekly_df["Symbol"].eq(symbol), "Weekly_Candle_Date"].iloc[0],
                "Setup_Interpretation": "EARLY_FLOW_IMPROVING" if (vfi_slope4 > 0 and vfi_slope8 > 0 and price_ret_20 <= 0.12 and dscore >= 50) else "WATCH / NEEDS_CONFIRMATION",
            })
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "DAILY", "Error": repr(exc)})
    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df = daily_df.sort_values("Final_Rank_Score", ascending=False).reset_index(drop=True)
        daily_df.insert(0, "Rank", range(1, len(daily_df) + 1))
    weekly_out = weekly_df.drop(columns=[], errors="ignore")
    weekly_out.to_csv(out / "us_100_weekly_screen.csv", index=False)
    daily_df.to_csv(out / "us_daily_shortlist_ranking.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps({
        "requested_symbols": len(symbols), "weekly_analyzed": int(weekly_df["Weekly_Screen_Status"].ne("ERROR").sum()) if "Weekly_Screen_Status" in weekly_df else 0,
        "weekly_candidates": len(rankable), "daily_analyzed": len(daily_df),
        "weekly_candle_policy": "completed Friday candles only",
        "daily_candle_policy": "completed daily candles only",
        "universe_note": "Curated liquid US-listed stocks and ETFs, not a live volume-ranked top-100 list.",
        "method_note": "Technical OHLCV proxies; not direct institutional-flow data. Final score is heuristic and requires out-of-sample backtesting.",
        "errors": errors,
    }, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"requested_symbols": len(symbols), "weekly_candidates": len(rankable),
                      "daily_analyzed": len(daily_df), "errors": len(errors)}, indent=2))
    if not daily_df.empty:
        print(daily_df.head(30).to_string(index=False))

if __name__ == "__main__":
    main()
