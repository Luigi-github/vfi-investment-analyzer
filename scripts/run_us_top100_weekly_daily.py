"""Weekly-first VFI screening of the 100 most liquid US-listed equities.

The universe is a broad candidate pool. The final 100 are selected dynamically
by median 20-session dollar volume, then filtered weekly and studied daily.
Only completed weekly bars are used for the first-stage decision.
"""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator

OUTPUT = ROOT / "reports" / "us_top100_weekly_daily"
LOOKBACK = "10y"
TOP_N = 100
MIN_DAILY_ROWS = 260

# Broad pool of highly traded US-listed stocks/ADRs. Selection of the actual
# 100 is data-driven by median 20-session dollar volume, not by this ordering.
CANDIDATES = list(dict.fromkeys("""
AAPL MSFT NVDA AMZN GOOGL GOOG META AVGO TSLA BRK-B JPM WMT LLY XOM
V MA MA UNH COST ORCL NFLX JNJ HD PG ABBV BAC CVX KO AMD CRM MRK
GE CAT IBM CSCO AMAT MU QCOM LIN WFC GS MS TMO ABT MCD PM RTX
AXP DIS TXN NOW ISRG INTU PLTR UBER LOW PEP COP C EG BMY PFE
SCHW BLK SPGI CHTR T DELL ANET PANW CRWD SNPS KLAC LRCX ADI
MDLZ UPS HON DE SYK BKNG TJX VZ CMCSA NKE AMGN GILD REGN
VRTX ADP MMC ICE CME USB PNC CARR ETN PH UNP FDX APH TT
SOFI COIN HOOD MARA MSTR RBLX SHOP SQ XYZ PYPL SNAP PINS
TGT CVS CI ELV HUM BSX ZTS DHR MDT EW NEM FCX NUE CMI
F SLB OXY EOG MPC PSX VLO KMI WMB LNG TFC COF APO AIG
ARM TSM ASML BABA PDD JD BIDU NVO SAP TM SONY
""".split()))

def latest_snapshot(data, indicator, timeframe):
    """Return the latest valid VFI snapshot and derived early-accumulation flags."""
    if data is None or data.empty:
        return None
    calc = indicator.calculate(data)
    needed = ["VFI", "VFI_Signal_20", "VFI_Signal_50"]
    valid = calc.dropna(subset=needed).copy()
    if len(valid) < 3:
        return None
    row = valid.iloc[-1]
    prev = valid.iloc[-2]
    # Slopes are measured over the most recent completed bar.
    vfi_slope = float(row["VFI"] - prev["VFI"])
    ema20_slope = float(row["VFI_Signal_20"] - prev["VFI_Signal_20"])
    ema50_slope = float(row["VFI_Signal_50"] - prev["VFI_Signal_50"])
    cross = bool(
        row["VFI_Signal_20"] > row["VFI_Signal_50"]
        and prev["VFI_Signal_20"] <= prev["VFI_Signal_50"]
    )
    eligible = bool(
        row["VFI"] < 0
        and row["VFI_Signal_20"] < 0
        and row["VFI_Signal_50"] < 0
        and vfi_slope > 0
        and ema20_slope > 0
        and row["VFI_Signal_20"] > row["VFI_Signal_50"]
    )
    return {
        "Timeframe": timeframe,
        "Signal_Date": valid.index[-1].date().isoformat(),
        "Close": float(row["Close"]),
        "VFI": float(row["VFI"]),
        "VFI_EMA20": float(row["VFI_Signal_20"]),
        "VFI_EMA50": float(row["VFI_Signal_50"]),
        "VFI_Slope": vfi_slope,
        "EMA20_Slope": ema20_slope,
        "EMA50_Slope": ema50_slope,
        "EMA20_Above_EMA50": bool(row["VFI_Signal_20"] > row["VFI_Signal_50"]),
        "Fresh_EMA20_Cross_Above_EMA50": cross,
        "VFI_Negative": bool(row["VFI"] < 0),
        "Signal_EMA20_Negative": bool(row["VFI_Signal_20"] < 0),
        "Signal_EMA50_Negative": bool(row["VFI_Signal_50"] < 0),
        "VFI_Slope_Bullish": bool(vfi_slope > 0),
        "EMA20_Slope_Bullish": bool(ema20_slope > 0),
        "Early_Accumulation_Eligible": eligible,
    }

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    market = MarketDataService()
    indicator = VFIIndicator(signal_length=20, signal_length_50=50)
    downloaded = {}
    errors = []
    print(f"Candidate pool: {len(CANDIDATES)} US-listed stocks/ADRs")
    print(f"Data period: {LOOKBACK}; target universe: top {TOP_N} by median 20-session dollar volume")

    for i, symbol in enumerate(CANDIDATES, 1):
        try:
            daily = market.get_daily_data(symbol, period=LOOKBACK)
            daily = daily.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(daily) < MIN_DAILY_ROWS:
                errors.append({"Symbol": symbol, "Error": f"INSUFFICIENT_HISTORY:{len(daily)}"})
                continue
            recent = daily.tail(20)
            dollar_volume = (recent["Close"] * recent["Volume"]).replace([np.inf, -np.inf], np.nan).dropna()
            median_dollar_volume = float(dollar_volume.median()) if not dollar_volume.empty else 0.0
            if median_dollar_volume <= 0:
                errors.append({"Symbol": symbol, "Error": "NO_RECENT_DOLLAR_VOLUME"})
                continue
            downloaded[symbol] = {
                "daily": daily,
                "median_dollar_volume": median_dollar_volume,
                "latest_daily_date": daily.index[-1].date().isoformat(),
            }
            print(f"[{i}/{len(CANDIDATES)}] OK {symbol}: median $ volume={median_dollar_volume:,.0f}")
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc)[:300]})
            print(f"[{i}/{len(CANDIDATES)}] ERROR {symbol}: {exc}")
        # Gentle pacing for Yahoo Finance; workflow runs in the cloud.
        time.sleep(0.05)

    if len(downloaded) < 50:
        raise RuntimeError(f"Only {len(downloaded)} symbols downloaded; refusing to publish an unreliable top-100 screen.")

    liquidity = pd.DataFrame([
        {"Symbol": s, "Median_Daily_Dollar_Volume_20D": d["median_dollar_volume"],
         "Latest_Daily_Date": d["latest_daily_date"], "Daily_Rows": len(d["daily"])}
        for s, d in downloaded.items()
    ]).sort_values("Median_Daily_Dollar_Volume_20D", ascending=False)
    top = liquidity.head(TOP_N).copy()
    top["Liquidity_Rank"] = range(1, len(top) + 1)
    top_symbols = top["Symbol"].tolist()
    top.to_csv(OUTPUT / "us_top100_liquidity.csv", index=False)
    print(f"Selected {len(top_symbols)} assets for the weekly-first stage.")

    weekly_rows = []
    for i, symbol in enumerate(top_symbols, 1):
        daily = downloaded[symbol]["daily"]
        weekly = market.convert_to_weekly(daily)
        snap = latest_snapshot(weekly, indicator, "WEEKLY")
        if snap is None:
            errors.append({"Symbol": symbol, "Error": "INSUFFICIENT_WEEKLY_VFI_HISTORY"})
            continue
        weekly_rows.append({
            "Symbol": symbol,
            "Liquidity_Rank": int(top.loc[top["Symbol"] == symbol, "Liquidity_Rank"].iloc[0]),
            "Median_Daily_Dollar_Volume_20D": downloaded[symbol]["median_dollar_volume"],
            "Weekly_Rows": len(weekly),
            **snap,
        })
        print(f"[weekly {i}/{len(top_symbols)}] {symbol}: VFI={snap['VFI']:.3f}, EMA20={snap['VFI_EMA20']:.3f}, EMA50={snap['VFI_EMA50']:.3f}, eligible={snap['Early_Accumulation_Eligible']}")

    weekly_df = pd.DataFrame(weekly_rows)
    if weekly_df.empty:
        raise RuntimeError("No weekly VFI snapshots could be produced.")
    weekly_df["VFI_Proximity_Score"] = 100.0 / (1.0 + weekly_df["VFI"].abs())
    weekly_df["VFI_Slope_Percentile"] = weekly_df["VFI_Slope"].rank(pct=True) * 100.0
    weekly_df["EMA20_Slope_Percentile"] = weekly_df["EMA20_Slope"].rank(pct=True) * 100.0
    weekly_df["Liquidity_Percentile"] = weekly_df["Median_Daily_Dollar_Volume_20D"].rank(pct=True) * 100.0
    weekly_df["Weekly_Score"] = (
        0.40 * weekly_df["VFI_Proximity_Score"]
        + 0.25 * weekly_df["VFI_Slope_Percentile"]
        + 0.20 * weekly_df["EMA20_Slope_Percentile"]
        + 0.15 * weekly_df["Liquidity_Percentile"]
    )
    weekly_df = weekly_df.sort_values(
        ["Early_Accumulation_Eligible", "Fresh_EMA20_Cross_Above_EMA50", "Weekly_Score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    weekly_df.insert(0, "Weekly_Rank", range(1, len(weekly_df) + 1))
    weekly_df.to_csv(OUTPUT / "weekly_screen_all_top100.csv", index=False)

    weekly_candidates = weekly_df[weekly_df["Early_Accumulation_Eligible"]].copy()
    weekly_candidates.to_csv(OUTPUT / "weekly_candidates.csv", index=False)
    print(f"Weekly stage: {len(weekly_candidates)} candidates passed all VFI/EMA conditions.")

    daily_rows = []
    for i, row in enumerate(weekly_candidates.itertuples(index=False), 1):
        symbol = row.Symbol
        daily = downloaded[symbol]["daily"]
        snap = latest_snapshot(daily, indicator, "DAILY")
        if snap is None:
            errors.append({"Symbol": symbol, "Error": "INSUFFICIENT_DAILY_VFI_HISTORY"})
            continue
        daily_rows.append({
            "Symbol": symbol,
            "Weekly_Rank": int(row.Weekly_Rank),
            "Weekly_VFI": float(row.VFI),
            "Weekly_EMA20": float(row.VFI_EMA20),
            "Weekly_EMA50": float(row.VFI_EMA50),
            "Weekly_VFI_Slope": float(row.VFI_Slope),
            "Weekly_EMA20_Slope": float(row.EMA20_Slope),
            "Weekly_Fresh_Cross": bool(row.Fresh_EMA20_Cross_Above_EMA50),
            "Median_Daily_Dollar_Volume_20D": downloaded[symbol]["median_dollar_volume"],
            **snap,
        })
        print(f"[daily {i}/{len(weekly_candidates)}] {symbol}: VFI={snap['VFI']:.3f}, EMA20={snap['VFI_EMA20']:.3f}, EMA50={snap['VFI_EMA50']:.3f}, daily eligible={snap['Early_Accumulation_Eligible']}")

    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df = daily_df.sort_values(
            ["Early_Accumulation_Eligible", "Fresh_EMA20_Cross_Above_EMA50", "Weekly_Rank"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        daily_df.insert(0, "Daily_Rank", range(1, len(daily_df) + 1))
    daily_df.to_csv(OUTPUT / "daily_confirmation_of_weekly_candidates.csv", index=False)
    daily_confirmed = daily_df[daily_df["Early_Accumulation_Eligible"]] if not daily_df.empty else daily_df
    daily_confirmed.to_csv(OUTPUT / "weekly_and_daily_confluence.csv", index=False)
    pd.DataFrame(errors).to_csv(OUTPUT / "download_errors.csv", index=False)

    summary = [
        ("Candidate_pool_size", len(CANDIDATES)),
        ("Downloaded_symbols", len(downloaded)),
        ("Liquidity_universe_size", len(top_symbols)),
        ("Weekly_eligible_candidates", len(weekly_candidates)),
        ("Daily_studied_from_weekly", len(daily_df)),
        ("Weekly_and_daily_confluence", len(daily_confirmed)),
        ("Download_or_analysis_errors", len(errors)),
        ("Weekly_signal_definition", "VFI<0; EMA20(VFI)<0; EMA50(VFI)<0; slope(VFI)>0; slope(EMA20)>0; EMA20(VFI)>EMA50(VFI)"),
        ("Daily_stage", "Only symbols passing weekly stage are analyzed daily"),
    ]
    pd.DataFrame(summary, columns=["Metric", "Value"]).to_csv(OUTPUT / "run_summary.csv", index=False)
    print("\n=== WEEKLY TOP RANKING (first 25) ===")
    cols = ["Weekly_Rank", "Symbol", "VFI", "VFI_EMA20", "VFI_EMA50", "VFI_Slope", "EMA20_Slope", "Fresh_EMA20_Cross_Above_EMA50", "Weekly_Score", "Early_Accumulation_Eligible"]
    print(weekly_df[cols].head(25).to_string(index=False))
    print("\n=== DAILY CONFIRMATION OF WEEKLY CANDIDATES ===")
    if daily_df.empty:
        print("No weekly candidates passed the filter; daily stage correctly has zero assets.")
    else:
        print(daily_df[["Daily_Rank", "Symbol", "VFI", "VFI_EMA20", "VFI_EMA50", "VFI_Slope", "EMA20_Slope", "Fresh_EMA20_Cross_Above_EMA50", "Early_Accumulation_Eligible"]].to_string(index=False))
    print(f"\nReports written to: {OUTPUT}")

if __name__ == "__main__":
    main()
