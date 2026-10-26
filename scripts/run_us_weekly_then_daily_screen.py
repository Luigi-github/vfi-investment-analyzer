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
ADBE ACN INTU NXPI MCHP ON MPWR ADI FTNT ZS OKTA MDB TEAM WDAY VEEV TTD EA TTWO EBAY ETSY
PGR CB TRV ALL MET PRU AFL AIG COF PNC TFC BK STT NTRS MTB RF KEY CFG FITB HBAN CMA
MO CL KMB GIS KHC KDP KR SYY DG DLTR ROST DRI DPZ ELV HUM DHR ZTS CARR EMR ITW CMI PWR
XEL ETR AEP PCG EIX WEC AES PPL ED EQIX CCI PSA VICI SPG FIS FI GPN VRSK PAYX NDAQ CBOE MCO MSCI
TEL GLW HPQ HPE JNPR MSI FSLR ENPH HAL BKR FANG DVN CTRA OKE EPD ET NOC GD HII LHX
CSX NSC CNI CP DAL UAL AAL CCL RCL EXPE MAR DELL ROKU WBD FOX PARA GRAB NTES TME SE NU
RBLX DKNG CAVA NTRA ALNY PODD IDXX ILMN IQV ZBH BDX HOLX COO WAT STE TECH RMD
NUE STLD PKG IP CF MOS FMC IFF ECL SHW PPG APD ETRN URI FAST GWW RSG WM
AON AJG BRO ACGL ERIE WRB L HIG CINF
""".split() + US_CANDIDATE_UNIVERSE))

def safe_cache_name(symbol: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in symbol)


def load_or_download_daily(service, symbol: str, period: str, cache_dir: Path, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Reuse locally persisted OHLCV data; download only on cache miss or explicit refresh."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{safe_cache_name(symbol)}.csv"
    if cache_file.exists() and not refresh:
        try:
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            cached.index = pd.to_datetime(cached.index)
            if not cached.empty and {"Open", "High", "Low", "Close", "Volume"}.issubset(cached.columns):
                return cached.sort_index(), "CACHE"
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    downloaded = service.get_daily_data(symbol, period=period)
    if not downloaded.empty:
        downloaded.to_csv(cache_file, index_label="Date")
    return downloaded, "DOWNLOADED"


def safe_slope(series, bars=4):
    s = pd.to_numeric(series, errors="coerce").dropna().tail(bars)
    if len(s) < bars:
        return float("nan")
    return float(np.polyfit(np.arange(len(s), dtype=float), s.to_numpy(dtype=float), 1)[0])


def vfi_ma_state(series):
    """VFI 20/50-bar moving-average state and the timing of crossover events."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 51:
        return {
            "VFI_MA20": float("nan"), "VFI_MA50": float("nan"),
            "VFI_MA20_Prev": float("nan"), "VFI_MA50_Prev": float("nan"),
            "VFI_MA20_Above_MA50": False, "VFI_MA_Bullish_Cross": False,
            "VFI_MA20_Zero_Cross_Recent": False, "VFI_MA50_Zero_Cross_Recent": False,
            "VFI_MA20_Zero_Cross_Bars_Ago": None, "VFI_MA50_Zero_Cross_Bars_Ago": None,
            "VFI_MA_Bullish_Cross_Bars_Ago": None,
            "VFI_MA20_Zero_Cross_Date": None, "VFI_MA50_Zero_Cross_Date": None,
            "VFI_MA_Bullish_Cross_Date": None,
            "VFI_MA_Phase": "INSUFFICIENT_HISTORY",
        }
    ma20 = s.rolling(20, min_periods=20).mean()
    ma50 = s.rolling(50, min_periods=50).mean()
    m20, m50 = float(ma20.iloc[-1]), float(ma50.iloc[-1])
    p20, p50 = float(ma20.iloc[-2]), float(ma50.iloc[-2])
    cross20 = (ma20.shift(1) < 0) & (ma20 >= 0)
    cross50 = (ma50.shift(1) < 0) & (ma50 >= 0)
    bullish_cross = (ma20.shift(1) <= ma50.shift(1)) & (ma20 > ma50)
    def last_event_age(events):
        positions = np.flatnonzero(events.fillna(False).to_numpy(dtype=bool))
        return int(len(s) - 1 - positions[-1]) if len(positions) else None
    def last_event_date(events):
        positions = np.flatnonzero(events.fillna(False).to_numpy(dtype=bool))
        return s.index[positions[-1]].strftime("%Y-%m-%d") if len(positions) else None
    age20, age50, age_cross = last_event_age(cross20), last_event_age(cross50), last_event_age(bullish_cross)
    ma20_zero = age20 == 0
    ma50_zero = age50 == 0
    cross_20_50 = age_cross == 0
    if m20 < 0 and m50 < 0 and m20 > m50:
        phase = "EARLY_SETUP_BOTH_NEGATIVE_MA20_ABOVE_MA50"
    elif m20 < 0 and m50 < 0:
        phase = "NEGATIVE_REGIME_WAITING_FOR_MA_CROSS"
    elif m20 >= 0 and m50 < 0:
        phase = "MA20_CROSSED_ZERO_MA50_STILL_NEGATIVE"
    elif m20 >= 0 and m50 >= 0:
        phase = "BOTH_MAS_ABOVE_ZERO"
    else:
        phase = "MIXED_OR_TURNING_DOWN"
    return {
        "VFI_MA20": m20, "VFI_MA50": m50,
        "VFI_MA20_Prev": p20, "VFI_MA50_Prev": p50,
        "VFI_MA20_Above_MA50": bool(m20 > m50),
        "VFI_MA_Bullish_Cross": cross_20_50,
        "VFI_MA20_Zero_Cross_Recent": ma20_zero,
        "VFI_MA50_Zero_Cross_Recent": ma50_zero,
        "VFI_MA20_Zero_Cross_Bars_Ago": age20,
        "VFI_MA50_Zero_Cross_Bars_Ago": age50,
        "VFI_MA_Bullish_Cross_Bars_Ago": age_cross,
        "VFI_MA20_Zero_Cross_Date": last_event_date(cross20),
        "VFI_MA50_Zero_Cross_Date": last_event_date(cross50),
        "VFI_MA_Bullish_Cross_Date": last_event_date(bullish_cross),
        "VFI_MA_Phase": phase,
    }


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
            "VFI_Signal": float(latest.get("VFI_Signal_20", latest.get("VFI_Signal", np.nan))),
            "VFI_Signal_20": float(latest.get("VFI_Signal_20", latest.get("VFI_Signal", np.nan))),
            "VFI_Signal_50": float(latest.get("VFI_Signal_50", np.nan)),
            "VFI_Histogram": float(latest.get("VFI_Histogram_20", latest.get("VFI_Histogram", np.nan))),
            "VFI_Histogram_20": float(latest.get("VFI_Histogram_20", latest.get("VFI_Histogram", np.nan))),
            "VFI_Histogram_50": float(latest.get("VFI_Histogram_50", np.nan)),
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
    # User's sequence: both VFI moving averages are negative, MA20 rises above
    # MA50 first, then zero-crosses are tracked as later confirmation.
    vfi_series = pd.to_numeric(VFIIndicator().calculate(weekly)["VFI"], errors="coerce").dropna()
    ma_state = vfi_ma_state(vfi_series)
    vfi_ma20 = ma_state["VFI_MA20"]
    vfi_ma50 = ma_state["VFI_MA50"]
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

    # User-specific opportunity score over the most recent 260 completed
    # weekly VFI observations. Positive points reward sustained recovery below
    # zero; negative points penalize distribution above zero. This scores all
    # liquid names rather than filtering the entire universe to qualifiers.
    score_series = vfi_series.tail(260)
    score_slope4 = (score_series - score_series.shift(4)) / 4.0
    score_slope8 = (score_series - score_series.shift(8)) / 8.0
    rising_below_zero = (score_series < 0) & (score_slope4 > 0) & (score_slope8 > 0)
    falling_above_zero = (score_series > 0) & (score_slope4 < 0) & (score_slope8 < 0)
    positive_points = int(rising_below_zero.sum())
    distribution_points = int(falling_above_zero.sum())

    ema5 = score_series.ewm(span=5, adjust=False, min_periods=5).mean()
    ema20 = score_series.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = score_series.ewm(span=50, adjust=False, min_periods=50).mean()
    ema5_now, ema20_now, ema50_now = float(ema5.iloc[-1]), float(ema20.iloc[-1]), float(ema50.iloc[-1])

    historical_median = float(score_series.median())
    historical_q95 = float(score_series.quantile(0.95))
    if historical_q95 > historical_median and value > historical_median:
        high_vfi_factor = float(np.clip((value - historical_median) / (historical_q95 - historical_median), 0.0, 1.0))
    else:
        high_vfi_factor = 0.0
    high_vfi_penalty = positive_points * high_vfi_factor
    all_emas_positive_penalty = 0.5 * positive_points if ema5_now > 0 and ema20_now > 0 and ema50_now > 0 else 0.0
    ema_inversion_penalty = (
        (0.5 * positive_points if ema5_now < ema20_now else 0.0)
        + (0.5 * positive_points if ema20_now < ema50_now else 0.0)
    )
    current_slope_penalty = (2.0 if slope8 < 0 else 0.0) + (1.0 if vfi.get("VFI_Slope_4B", np.nan) < 0 else 0.0)
    opportunity_score = (
        positive_points - distribution_points - high_vfi_penalty
        - all_emas_positive_penalty - ema_inversion_penalty - current_slope_penalty
    )

    price_return_13w = float(weekly["Close"].iloc[-1] / weekly["Close"].iloc[-14] - 1)
    # Early weekly setup: MA20 is above MA50 while BOTH remain below zero.
    # Do not wait for the current VFI or its averages to cross zero; track those
    # crosses as subsequent events so the scan can catch the move earlier.
    qualifies = bool(
        np.isfinite(value) and value < 0
        and np.isfinite(vfi_ma50) and vfi_ma50 < 0
        and np.isfinite(vfi_ma20) and vfi_ma20 < 0
        and ma_state["VFI_MA20_Above_MA50"]
        and np.isfinite(vfi.get("VFI_Slope_4B", np.nan)) and vfi["VFI_Slope_4B"] > 0
        and np.isfinite(slope8) and slope8 > 0
        and monotonic4 and np.isfinite(r2_4) and r2_4 >= 0.65
        and np.isfinite(price_return_13w) and price_return_13w <= 0.10
    )
    # Rank by MA20 proximity to zero, positive MA spread, then slope quality.
    ma_spread = vfi_ma20 - vfi_ma50 if np.isfinite(vfi_ma20) and np.isfinite(vfi_ma50) else float("nan")
    score = float(opportunity_score)
    return {
        "Symbol": symbol,
        "Weekly_Screen_Status": "CANDIDATE" if qualifies else "NOT_SELECTED",
        "Weekly_Score": score,
        "Weekly_Opportunity_Score": score,
        "Weekly_Negative_Uptrend_Weeks_260W": positive_points,
        "Weekly_Positive_Downtrend_Weeks_260W": distribution_points,
        "Weekly_High_VFI_Penalty": high_vfi_penalty,
        "Weekly_All_EMAs_Positive_Penalty": all_emas_positive_penalty,
        "Weekly_EMA_Inversion_Penalty": ema_inversion_penalty,
        "Weekly_EMA5_VFI": ema5_now,
        "Weekly_EMA20_VFI": ema20_now,
        "Weekly_EMA50_VFI": ema50_now,
        "Weekly_VFI": value,
        "Weekly_VFI_MA50": vfi_ma50,
        "Weekly_VFI_MA20": vfi_ma20,
        "Weekly_VFI_MA20_Above_MA50": ma_state["VFI_MA20_Above_MA50"],
        "Weekly_VFI_MA_Bullish_Cross": ma_state["VFI_MA_Bullish_Cross"],
        "Weekly_VFI_MA20_Zero_Cross_Recent": ma_state["VFI_MA20_Zero_Cross_Recent"],
        "Weekly_VFI_MA50_Zero_Cross_Recent": ma_state["VFI_MA50_Zero_Cross_Recent"],
        "Weekly_VFI_MA20_Zero_Cross_Bars_Ago": ma_state["VFI_MA20_Zero_Cross_Bars_Ago"],
        "Weekly_VFI_MA50_Zero_Cross_Bars_Ago": ma_state["VFI_MA50_Zero_Cross_Bars_Ago"],
        "Weekly_VFI_MA_Bullish_Cross_Bars_Ago": ma_state["VFI_MA_Bullish_Cross_Bars_Ago"],
        "Weekly_VFI_MA20_Zero_Cross_Date": ma_state["VFI_MA20_Zero_Cross_Date"],
        "Weekly_VFI_MA50_Zero_Cross_Date": ma_state["VFI_MA50_Zero_Cross_Date"],
        "Weekly_VFI_MA_Bullish_Cross_Date": ma_state["VFI_MA_Bullish_Cross_Date"],
        "Weekly_VFI_MA_Phase": ma_state["VFI_MA_Phase"],
        "Weekly_VFI_MA_Spread": ma_spread,
        "Weekly_VFI_Prev": vfi.get("VFI_Prev"),
        "Weekly_VFI_Signal": vfi.get("VFI_Signal_20", vfi.get("VFI_Signal")),
        "Weekly_VFI_Signal_20": vfi.get("VFI_Signal_20", vfi.get("VFI_Signal")),
        "Weekly_VFI_Signal_50": vfi.get("VFI_Signal_50"),
        "Weekly_VFI_Histogram": vfi.get("VFI_Histogram_20", vfi.get("VFI_Histogram")),
        "Weekly_VFI_Histogram_20": vfi.get("VFI_Histogram_20", vfi.get("VFI_Histogram")),
        "Weekly_VFI_Histogram_50": vfi.get("VFI_Histogram_50"),
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
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--max-daily-candidates", type=int, default=5)
    p.add_argument("--output-dir", default="reports/us_weekly_then_daily")
    p.add_argument("--cache-dir", default="data/market_cache/us_equities")
    p.add_argument("--refresh-cache", action="store_true", help="Force fresh Yahoo Finance downloads and replace cached OHLCV files.")
    args = p.parse_args()
    symbols = list(dict.fromkeys(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    service = MarketDataService()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / args.cache_dir
    raw_data, liquidity_rows, errors = {}, [], []
    cache_hits = 0
    downloaded_count = 0
    print(f"Downloading daily history for {len(symbols)} US-listed symbols to rank liquidity...", flush=True)
    for i, symbol in enumerate(symbols, 1):
        print(f"[LIQUIDITY {i}/{len(symbols)}] {symbol}", flush=True)
        try:
            raw_daily, cache_status = load_or_download_daily(
                service, symbol, period=args.period, cache_dir=cache_dir,
                refresh=args.refresh_cache,
            )
            cache_hits += cache_status == "CACHE"
            downloaded_count += cache_status == "DOWNLOADED"
            daily = completed_daily(raw_daily)
            if daily.empty or not {"Close", "Volume"}.issubset(daily.columns):
                raise ValueError("No usable OHLCV data")
            # VFI(130) needs about 160 weekly bars to produce values; the
            # 50-week VFI trend needs roughly 210 weekly bars to warm up.
            weekly_history = to_weekly(daily)
            if len(weekly_history) < 210:
                message = f"INSUFFICIENT_WEEKLY_HISTORY:{len(weekly_history)}_bars; need >=210"
                errors.append({"Symbol": symbol, "Stage": "LIQUIDITY", "Error": message})
                liquidity_rows.append({
                    "Symbol": symbol, "Average_Dollar_Turnover_20D": np.nan,
                    "Liquidity_Data_Date": daily.index[-1].strftime("%Y-%m-%d"),
                    "Liquidity_Status": message,
                })
                continue
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
    weekly_df = weekly_df.sort_values(
        ["Weekly_Opportunity_Score", "Weekly_Screen_Status", "Average_Dollar_Turnover_20D"],
        ascending=[False, True, False], na_position="last"
    ).reset_index(drop=True)
    weekly_df.insert(0, "Opportunity_Rank", range(1, len(weekly_df) + 1))
    weekly_df.to_csv(out / f"us_top_{args.top}_weekly_screen.csv", index=False)
    liquidity_df.sort_values("Average_Dollar_Turnover_20D", ascending=False).to_csv(out / f"us_top_{args.top}_liquidity_universe.csv", index=False)

    candidates = weekly_df[weekly_df["Weekly_Screen_Status"].eq("CANDIDATE")].copy()
    for col in ["Weekly_VFI", "Weekly_VFI_Slope_20W", "Weekly_VFI_Slope_50W", "Weekly_Score"]:
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    candidates["VFI_Distance_To_Zero"] = candidates["Weekly_VFI"].abs()
    candidates = candidates.sort_values(
        ["Weekly_Opportunity_Score", "Weekly_VFI_MA20", "Weekly_VFI_MA_Spread",
         "Weekly_VFI_Slope_20W", "Weekly_VFI_Slope_50W", "Weekly_VFI_R2_4W"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    candidates.insert(0, "Rank", range(1, len(candidates) + 1))
    candidates.to_csv(out / "us_weekly_vfi_candidates.csv", index=False)
    print("\nWEEKLY VFI CANDIDATES: VFI MA20 > MA50, both averages below zero, rising VFI, price not extended", flush=True)
    print(candidates.to_string(index=False) if not candidates.empty else "No assets passed the exact weekly VFI filter.", flush=True)

    # Stage 2 is strictly conditional: only assets passing the completed-weekly
    # filter are eligible for daily confirmation. Never substitute near-misses.
    daily_rows = []
    weekly_rank_map = {
        row.Symbol: int(row.Rank)
        for row in candidates[["Symbol", "Rank"]].itertuples(index=False)
    } if not candidates.empty else {}
    followup_symbols = candidates["Symbol"].head(max(0, args.max_daily_candidates)).tolist()
    followup_mode = "STRICT_WEEKLY_CANDIDATES_ONLY"
    print(f"\\nDAILY FOLLOW-UP for {len(followup_symbols)} weekly-qualified assets ({followup_mode})", flush=True)
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
            daily_series = pd.to_numeric(VFIIndicator().calculate(daily)["VFI"], errors="coerce").dropna()
            daily_ma = vfi_ma_state(daily_series)
            # Daily confirmation is only run on the weekly shortlist. Prefer the
            # same early setup: MA20 > MA50 while both are negative, before zero-cross.
            early = bool(
                daily_ma["VFI_MA20"] < 0 and daily_ma["VFI_MA50"] < 0
                and daily_ma["VFI_MA20_Above_MA50"]
                and np.isfinite(slope4) and slope4 > 0
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
                "Weekly_Rank": int(weekly_rank_map[symbol]),
                "Daily_Early_Accumulation_Candidate": early,
                "Daily_Score": daily_score,
                "Daily_Candle_Date": daily.index[-1].strftime("%Y-%m-%d"),
                "Daily_Price": float(close.iloc[-1]),
                "Daily_Price_Return_20B": ret20,
                "Daily_VFI": daily_vfi,
                "Daily_VFI_Signal_20": vfi.get("VFI_Signal_20", vfi.get("VFI_Signal")),
                "Daily_VFI_Signal_50": vfi.get("VFI_Signal_50"),
                "Daily_VFI_Histogram_20": vfi.get("VFI_Histogram_20", vfi.get("VFI_Histogram")),
                "Daily_VFI_Histogram_50": vfi.get("VFI_Histogram_50"),
                "Daily_VFI_MA20": daily_ma["VFI_MA20"],
                "Daily_VFI_MA50": daily_ma["VFI_MA50"],
                "Daily_VFI_MA20_Above_MA50": daily_ma["VFI_MA20_Above_MA50"],
                "Daily_VFI_MA_Bullish_Cross": daily_ma["VFI_MA_Bullish_Cross"],
                "Daily_VFI_MA20_Zero_Cross_Recent": daily_ma["VFI_MA20_Zero_Cross_Recent"],
                "Daily_VFI_MA50_Zero_Cross_Recent": daily_ma["VFI_MA50_Zero_Cross_Recent"],
                "Daily_VFI_MA20_Zero_Cross_Bars_Ago": daily_ma["VFI_MA20_Zero_Cross_Bars_Ago"],
                "Daily_VFI_MA50_Zero_Cross_Bars_Ago": daily_ma["VFI_MA50_Zero_Cross_Bars_Ago"],
                "Daily_VFI_MA_Bullish_Cross_Bars_Ago": daily_ma["VFI_MA_Bullish_Cross_Bars_Ago"],
                "Daily_VFI_MA20_Zero_Cross_Date": daily_ma["VFI_MA20_Zero_Cross_Date"],
                "Daily_VFI_MA50_Zero_Cross_Date": daily_ma["VFI_MA50_Zero_Cross_Date"],
                "Daily_VFI_MA_Bullish_Cross_Date": daily_ma["VFI_MA_Bullish_Cross_Date"],
                "Daily_VFI_MA_Phase": daily_ma["VFI_MA_Phase"],
                "Daily_VFI_MA_Spread": daily_ma["VFI_MA20"] - daily_ma["VFI_MA50"],
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
        "weekly_selected_for_analysis": len(selected_symbols),
        "ohlcv_cache_directory": str(cache_dir.relative_to(ROOT)),
        "ohlcv_cache_hits": int(cache_hits),
        "ohlcv_downloads": int(downloaded_count),
        "ohlcv_cache_policy": "OHLCV CSV files are reused across runs; pass --refresh-cache to replace them with fresh Yahoo Finance history.",
        "weekly_analyzed": int(weekly_df["Weekly_Screen_Status"].ne("ERROR").sum()) if not weekly_df.empty else 0,
        "weekly_vfi_candidates": int(len(candidates)),
        "daily_followup_analyzed": int(len(daily_df)),
        "daily_followup_mode": followup_mode,
        "daily_early_accumulation_candidates": int(daily_df.get("Daily_Early_Accumulation_Candidate", pd.Series(dtype=bool)).eq(True).sum()),
        "weekly_rule": "completed weekly candles; all eligible assets receive an opportunity score based on weeks with VFI below zero and positive 4W/8W slopes, less weeks with VFI above zero and negative slopes; proportional penalty when current VFI is high versus its own recent history; penalties when all VFI EMAs are positive or EMA ordering inverts. A strict early-accumulation candidate additionally requires VFI<0, VFI MA20>MA50 with both below zero, positive 4W/8W slopes, last 4 VFI values strictly rising with R2>=0.65 and 13W price return<=+10%. At least 210 completed weekly bars are required.",
        "daily_rule": "only strict weekly candidates are followed up on completed daily candles; if there are none, no daily analysis is performed; daily VFI MA20 > MA50 while BOTH averages remain below zero, positive 4D/8D VFI slopes, accumulation/distribution composite >=50 and 20-session price return <= +10%; zero-cross flags are reported; score is a heuristic, not a probability",
        "weekly_candle_policy": "completed weekly candles only; incomplete current week excluded",
        "universe_note": "Top N by mean Close*Volume over 20 completed sessions within an expanded curated US-listed common-stock/ADR seed universe; ETFs are excluded; at least 210 completed weekly bars are required before liquidity ranking. This is a reproducible universe screen, not a claim to rank every US listing.",
        "method_note": "VFI-only selection. OHLCV-based volume indicator; not direct institutional-flow data.",
        "errors": errors,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nRUN SUMMARY\n" + json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

# Execution trigger: run the weekly-first US liquidity and early-accumulation screen.
