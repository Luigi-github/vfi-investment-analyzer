"""Relative, weekly VFI early-accumulation ranking.

All EMA curves are calculated on the VFI series (never on price). Historical
percentiles are asset-relative and use only observations available up to the
current completed weekly candle. This is a research heuristic, not a forecast.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM
from scripts.run_us_weekly_then_daily_screen import US_LIQUID_UNIVERSE

EMA_PERIODS = {"ema5": 5, "ema20": 20, "ema80": 80, "ema200": 200}
EMA_WEIGHTS = {"ema5": 0.05, "ema20": 0.15, "ema80": 0.30, "ema200": 0.50}
LOOKBACK = 260
STREAK_CAP = 52


def trailing_true_streak(flags) -> int:
    """Return the current uninterrupted True run; previous runs do not count."""
    values = pd.Series(flags).fillna(False).astype(bool).to_numpy()
    count = 0
    for flag in values[::-1]:
        if not flag:
            break
        count += 1
    return count


def historical_percentile(history: pd.Series, current: float, lookback: int = LOOKBACK) -> float:
    """Percentile rank of current against the trailing history; no future data."""
    values = pd.to_numeric(history, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().tail(lookback)
    if not np.isfinite(current) or values.empty:
        return float("nan")
    # Mid-rank makes ties stable and bounds the result to [0, 100].
    less = float((values < current).sum())
    equal = float((values == current).sum())
    return 100.0 * (less + 0.5 * equal) / len(values)


def linear_r2(values) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 4 or not np.isfinite(arr).all():
        return 0.0
    x = np.arange(len(arr), dtype=float)
    fit = np.polyval(np.polyfit(x, arr, 1), x)
    total = float(np.sum((arr - arr.mean()) ** 2))
    if total <= 0:
        return 0.0
    return float(np.clip(1.0 - np.sum((arr - fit) ** 2) / total, 0.0, 1.0))


def regression_slope(values) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2 or not np.isfinite(arr).all():
        return float("nan")
    return float(np.polyfit(np.arange(len(arr), dtype=float), arr, 1)[0])


def score_weekly_vfi(vfi: pd.Series) -> dict:
    """EMA-only score on weekly VFI; raw VFI is used only to construct EMA curves."""
    s = pd.to_numeric(vfi, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 200:
        raise ValueError(f"Insufficient valid weekly VFI history ({len(s)}; need >=200 for EMA200)")

    emas = {
        key: s.ewm(span=period, adjust=False, min_periods=period).mean()
        for key, period in EMA_PERIODS.items()
    }
    ema_now = {key: float(series.iloc[-1]) for key, series in emas.items()}
    ema_streaks = {key: trailing_true_streak(series.diff() > 0) for key, series in emas.items()}
    ema_percentiles = {
        key: historical_percentile(series.iloc[:-1], ema_now[key], lookback=260)
        for key, series in emas.items()
    }

    # Signed position: negative EMA values are rewarded more when unusually
    # depressed versus their own history; positive values are penalized more
    # when unusually elevated. EMA timeframe weights remain 5/15/30/50.
    signed_components = {}
    for key in EMA_PERIODS:
        pct, value = ema_percentiles[key], ema_now[key]
        if not np.isfinite(pct):
            pct = 50.0
        signed_components[key] = (100.0 - pct) if value < 0 else (-pct if value > 0 else 0.0)
    signed_raw = sum(EMA_WEIGHTS[k] * signed_components[k] for k in EMA_PERIODS)
    ema_level_score = float(np.clip(50.0 + 0.5 * signed_raw, 0.0, 100.0))

    # Persistence is the weighted uninterrupted rising duration of each EMA.
    ema_persistence_score = sum(
        100.0 * EMA_WEIGHTS[k] * min(ema_streaks[k], STREAK_CAP) / STREAK_CAP
        for k in EMA_PERIODS
    )

    # EMA acceleration is measured on each EMA curve, not on price or raw VFI.
    acceleration_percentiles = {}
    ema_acceleration_values = {}
    ema_last_slopes = {}
    for key, series in emas.items():
        slope4_series = series.rolling(4).apply(regression_slope, raw=True)
        acceleration = slope4_series.diff(4)
        current_acceleration = float(acceleration.iloc[-1]) if len(acceleration) else float("nan")
        pct = historical_percentile(acceleration.iloc[:-1], current_acceleration, lookback=260)
        if not np.isfinite(pct):
            pct = 50.0
        current_slope = float(slope4_series.iloc[-1]) if len(slope4_series) else float("nan")
        # Positive acceleration is useful only when the EMA is actually rising.
        if not np.isfinite(current_slope) or current_slope <= 0:
            pct *= 0.50
        acceleration_percentiles[key] = pct
        ema_acceleration_values[key] = current_acceleration
        ema_last_slopes[key] = current_slope
    acceleration_score = float(sum(EMA_WEIGHTS[k] * acceleration_percentiles[k] for k in EMA_PERIODS))

    # Early accumulation rewards EMAs that remain below zero while beginning
    # a sustained rise. No near-zero threshold or raw VFI level is used.
    early_components = {}
    for key in EMA_PERIODS:
        if ema_now[key] < 0 and np.isfinite(ema_last_slopes[key]) and ema_last_slopes[key] > 0:
            duration = min(ema_streaks[key], 13) / 13.0
            acceleration = acceleration_percentiles[key] / 100.0
            early_components[key] = 100.0 * (0.60 * duration + 0.40 * acceleration)
        else:
            early_components[key] = 0.0
    early_accumulation_score = float(sum(EMA_WEIGHTS[k] * early_components[k] for k in EMA_PERIODS))

    gross_score = float(
        0.45 * ema_persistence_score
        + 0.30 * ema_level_score
        + 0.20 * acceleration_score
        + 0.05 * early_accumulation_score
    )

    # Deterioration penalty: a falling long EMA costs more than a falling short EMA.
    ema_negative_slope_weight = sum(
        EMA_WEIGHTS[k] for k in EMA_PERIODS
        if np.isfinite(ema_last_slopes[k]) and ema_last_slopes[k] < 0
    )
    ema_pullback_penalty = 0.25 * gross_score * ema_negative_slope_weight
    ema_structure_penalty = 0.0
    if ema_now["ema5"] < ema_now["ema20"]:
        ema_structure_penalty += 2.0
    if ema_now["ema20"] < ema_now["ema80"]:
        ema_structure_penalty += 4.0
    if ema_now["ema80"] < ema_now["ema200"]:
        ema_structure_penalty += 6.0
    all_positive_penalty = 8.0 if all(value > 0 for value in ema_now.values()) else 0.0
    total_penalty = min(
        ema_pullback_penalty + ema_structure_penalty + all_positive_penalty,
        0.55 * gross_score,
    )
    final_score = float(np.clip(gross_score - total_penalty, 0.0, 100.0))

    # VFI raw-value diagnostics are exported for interpretation only; they do
    # not enter Gross_Score, penalties, or Final_Score.
    current = float(s.iloc[-1])
    vfi_pct = historical_percentile(s.iloc[:-1], current, lookback=260)
    abs_pct = historical_percentile(s.iloc[:-1].abs(), abs(current), lookback=260)
    near_zero_diagnostic = float(np.clip(100.0 - abs_pct, 0.0, 100.0)) if np.isfinite(abs_pct) else float("nan")
    vfi_slope4 = regression_slope(s.tail(4).to_numpy())
    vfi_slope8 = regression_slope(s.tail(8).to_numpy())
    vfi_slope13 = regression_slope(s.tail(13).to_numpy())
    vfi_slope26 = regression_slope(s.tail(26).to_numpy())
    vfi_slope4_series = s.rolling(4).apply(regression_slope, raw=True)
    vfi_acceleration = vfi_slope4_series.diff(4)
    vfi_acceleration_value = float(vfi_acceleration.iloc[-1]) if len(vfi_acceleration) else float("nan")
    vfi_acceleration_pct = historical_percentile(vfi_acceleration.iloc[:-1], vfi_acceleration_value, lookback=260)
    if not np.isfinite(vfi_acceleration_pct):
        vfi_acceleration_pct = 50.0

    return {
        "VFI": current,
        "VFI_Historical_Percentile_DIAGNOSTIC": vfi_pct,
        "VFI_Near_Zero_Score_DIAGNOSTIC": near_zero_diagnostic,
        "VFI_Absolute_Magnitude_Percentile": abs_pct,
        "VFI_Near_Zero_Contribution": 0.0,
        "VFI_Slope_4W_DIAGNOSTIC": vfi_slope4,
        "VFI_Slope_8W_DIAGNOSTIC": vfi_slope8,
        "VFI_Slope_13W_DIAGNOSTIC": vfi_slope13,
        "VFI_Slope_26W_DIAGNOSTIC": vfi_slope26,
        "VFI_Slope_8W": vfi_slope8,
        "VFI_Slope_26W": vfi_slope26,
        "VFI_Acceleration_4W_DIAGNOSTIC": vfi_acceleration_value,
        "VFI_Acceleration_Percentile": vfi_acceleration_pct,
        "EMA_Persistence_Score": float(ema_persistence_score),
        "EMA_Level_Signed_Score": ema_level_score,
        "EMA_Level_Signed_Raw_Score": float(signed_raw),
        "EMA_Acceleration_Score": acceleration_score,
        "Acceleration_Score": acceleration_score,
        "Early_Accumulation_Score": early_accumulation_score,
        "Relative_Position_Score": ema_level_score,
        "EMA_Level_Contribution": 0.30 * ema_level_score,
        "VFI_Persistence_Score": 0.0,
        "Gross_Score": gross_score,
        "VFI_Positive_Extension_Penalty": 0.0,
        "All_EMAs_Positive_Penalty": all_positive_penalty,
        "EMA_Inversion_Penalty": ema_structure_penalty,
        "Pullback_Penalty": ema_pullback_penalty,
        "EMA_Pullback_Penalty": ema_pullback_penalty,
        "EMA_Negative_Slope_Weight": ema_negative_slope_weight,
        "Total_Penalty": total_penalty,
        "Final_Score": final_score,
        **{f"{key.upper()}_Signed_Level_Component": signed_components[key] for key in EMA_PERIODS},
        **{f"{key.upper()}_Value": value for key, value in ema_now.items()},
        **{f"{key.upper()}_Percentile": (ema_percentiles[key] if np.isfinite(ema_percentiles[key]) else 50.0) for key in EMA_PERIODS},
        **{f"{key.upper()}_Rising_Streak_Weeks": value for key, value in ema_streaks.items()},
        **{f"{key.upper()}_Acceleration_Percentile": acceleration_percentiles[key] for key in EMA_PERIODS},
        **{f"{key.upper()}_Acceleration_Value": ema_acceleration_values[key] for key in EMA_PERIODS},
        **{f"{key.upper()}_Slope_4W": ema_last_slopes[key] for key in EMA_PERIODS},
    }


def fetch_completed_weekly(symbol: str, period: str = "max") -> tuple[pd.DataFrame, dict]:
    """Fetch Yahoo's native 1wk OHLCV bars; never request or resample daily bars."""
    ticker = yf.Ticker(symbol)
    frame = ticker.history(
        period=period,
        interval="1wk",
        auto_adjust=False,
        actions=False,
        repair=False,
        keepna=True,
    )
    if frame is None or frame.empty:
        raise ValueError("Yahoo returned no native weekly bars")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns={str(col): str(col).title() for col in frame.columns})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Yahoo weekly response missing columns: {missing}")
    frame = frame[required].copy()
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    # Yahoo weekly timestamps can label the bar by its first session. Map each
    # provider bar to its Friday-ending exchange week, then remove the still
    # incomplete current week (including Thanksgiving's short week).
    week_ends = idx.to_period("W-FRI").end_time.normalize()
    frame.index = week_ends
    frame.index.name = "Week_End"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    current_week_end = today.to_period("W-FRI").end_time.normalize()
    frame = frame.loc[frame.index < current_week_end].copy()
    frame = frame.dropna(subset=["Close"])
    if frame.empty:
        raise ValueError("No completed weekly Yahoo bars after excluding the open week")

    # Preserve missing weeks without fabricating volume: carry forward only
    # the prior close as flat OHLC and set volume to zero. Keep an audit flag.
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="W-FRI", name="Week_End")
    original_dates = set(frame.index)
    frame = frame.reindex(full_index)
    synthetic = ~frame.index.isin(original_dates)
    previous_close = frame["Close"].ffill()
    for col in ("Open", "High", "Low", "Close"):
        frame[col] = frame[col].fillna(previous_close)
    frame["Volume"] = frame["Volume"].fillna(0.0)
    frame["Synthetic_Bar"] = synthetic
    frame = frame.dropna(subset=["Close"])
    frame[required] = frame[required].apply(pd.to_numeric, errors="coerce")
    if len(frame) < 330:
        raise ValueError(f"Only {len(frame)} completed weekly bars; need at least 330 to warm up VFI(130) and EMA200")
    metadata = {
        "Weekly_Candle_Date": frame.index[-1].strftime("%Y-%m-%d"),
        "Weekly_Bars": int(len(frame)),
        "Synthetic_Weekly_Bars": int(frame["Synthetic_Bar"].sum()),
        "Yahoo_Interval": "1wk",
        "Source": "Yahoo Finance native weekly OHLCV",
    }
    return frame, metadata



def monthly_from_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Approximate calendar-month OHLCV using only native Yahoo weekly bars."""
    monthly = weekly.resample("ME").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open", "High", "Low", "Close"])
    current_month_start = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize().replace(day=1)
    return monthly.loc[monthly.index < current_month_start].copy()


def score_monthly_divergence(series: pd.Series) -> dict:
    """EMA-only monthly divergence score; weights 5/20/80 = 10/30/60."""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 80:
        raise ValueError(f"Only {len(s)} valid monthly divergence values; need >=80 for EMA80")
    periods = {"ema5": 5, "ema20": 20, "ema80": 80}
    weights = {"ema5": 0.10, "ema20": 0.30, "ema80": 0.60}
    emas = {k: s.ewm(span=p, adjust=False, min_periods=p).mean() for k, p in periods.items()}
    values = {k: float(v.iloc[-1]) for k, v in emas.items()}
    streaks = {k: trailing_true_streak(v.diff() > 0) for k, v in emas.items()}
    persistence = sum(100.0 * weights[k] * min(streaks[k], 52) / 52.0 for k in periods)
    percentiles = {k: historical_percentile(v.iloc[:-1], values[k], lookback=240) for k, v in emas.items()}
    percentiles = {k: (pct if np.isfinite(pct) else 50.0) for k, pct in percentiles.items()}
    components = {}
    for k in periods:
        pct, value = percentiles[k], values[k]
        components[k] = (100.0 - pct) if value < 0 else (-pct if value > 0 else 0.0)
    raw_level = sum(weights[k] * components[k] for k in periods)
    level_score = float(np.clip(50.0 + 0.5 * raw_level, 0.0, 100.0))
    score = float(np.clip(0.60 * persistence + 0.40 * level_score, 0.0, 100.0))
    return {
        "Monthly_Divergence_Value": float(s.iloc[-1]),
        "Monthly_Divergence_EMA_Persistence_Score": float(persistence),
        "Monthly_Divergence_EMA_Level_Score": level_score,
        "Monthly_Divergence_Final_Score": score,
        **{f"Monthly_{k.upper()}_Value": values[k] for k in periods},
        **{f"Monthly_{k.upper()}_Percentile": percentiles[k] for k in periods},
        **{f"Monthly_{k.upper()}_Rising_Streak_Months": streaks[k] for k in periods},
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Rank US stocks by relative early accumulation in native Yahoo weekly VFI.")
    parser.add_argument("--symbols", default=",".join(US_LIQUID_UNIVERSE))
    parser.add_argument("--period", default="max")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--output-dir", default="reports/us_relative_vfi_ranking")
    parser.add_argument("--refresh-cache", action="store_true", help="Compatibility flag; Yahoo native weekly bars are fetched fresh on every run.")
    args = parser.parse_args()

    symbols = list(dict.fromkeys(x.strip().upper() for x in args.symbols.split(",") if x.strip()))
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    weekly_data: dict[str, pd.DataFrame] = {}
    liquidity_rows, errors = [], []

    def load_one(symbol: str):
        try:
            weekly, meta = fetch_completed_weekly(symbol, args.period)
            turnover = (
                pd.to_numeric(weekly["Close"], errors="coerce")
                * pd.to_numeric(weekly["Volume"], errors="coerce")
            ).replace([np.inf, -np.inf], np.nan).dropna().tail(4)
            dollar_turnover = float(turnover.mean()) if len(turnover) >= 3 else float("nan")
            if not np.isfinite(dollar_turnover):
                raise ValueError("Insufficient recent completed weekly dollar-volume observations")
            vfi_frame = VFIIndicator().calculate(weekly)
            valid_vfi = pd.to_numeric(vfi_frame["VFI"], errors="coerce").dropna()
            if len(valid_vfi) < 200:
                raise ValueError(f"Only {len(valid_vfi)} valid weekly VFI bars; need >=200 for a fully warmed EMA200")
            return symbol, weekly, {
                "Symbol": symbol,
                "Average_Dollar_Turnover_4W": dollar_turnover,
                "Data_Date": meta["Weekly_Candle_Date"],
                "Weekly_Bars": meta["Weekly_Bars"],
                "Synthetic_Weekly_Bars": meta["Synthetic_Weekly_Bars"],
                "Yahoo_Interval": "1wk",
                "Data_Source": meta["Source"],
                "Status": "OK",
            }, None
        except Exception as exc:
            return symbol, None, {
                "Symbol": symbol, "Average_Dollar_Turnover_4W": np.nan,
                "Data_Date": "", "Weekly_Bars": 0, "Synthetic_Weekly_Bars": 0,
                "Yahoo_Interval": "1wk", "Data_Source": "Yahoo Finance native weekly OHLCV",
                "Status": f"ERROR: {exc!r}",
            }, {"Symbol": symbol, "Stage": "WEEKLY_DATA", "Error": repr(exc)}

    print(f"Fetching native weekly Yahoo OHLCV for {len(symbols)} symbols (6 concurrent workers)...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(load_one, symbols))
    for i, (symbol, weekly, liquidity_row, error) in enumerate(results, 1):
        print(f"[WEEKLY YAHOO {i}/{len(symbols)}] {symbol}: {liquidity_row['Status']}", flush=True)
        if weekly is not None:
            weekly_data[symbol] = weekly
        liquidity_rows.append(liquidity_row)
        if error is not None:
            errors.append(error)

    liquidity_df = pd.DataFrame(liquidity_rows)
    eligible = liquidity_df[
        liquidity_df["Status"].eq("OK")
        & np.isfinite(pd.to_numeric(liquidity_df["Average_Dollar_Turnover_4W"], errors="coerce"))
    ]
    selected = eligible.sort_values("Average_Dollar_Turnover_4W", ascending=False).head(args.top)
    rows = []
    for i, item in enumerate(selected.itertuples(index=False), 1):
        symbol = item.Symbol
        print(f"[SCORE {i}/{len(selected)}] {symbol}", flush=True)
        try:
            weekly = weekly_data[symbol]
            vfi_frame = VFIIndicator().calculate(weekly)
            vfi = pd.to_numeric(vfi_frame["VFI"], errors="coerce").dropna()
            scored = score_weekly_vfi(vfi)
            scored.update({
                "Symbol": symbol,
                "Liquidity_Rank": i,
                "Average_Dollar_Turnover_4W": float(item.Average_Dollar_Turnover_4W),
                "Weekly_Candle_Date": vfi.index[-1].strftime("%Y-%m-%d"),
                "Weekly_Bars": len(weekly),
                "Synthetic_Weekly_Bars": int(weekly["Synthetic_Bar"].sum()),
                "Yahoo_Interval": "1wk",
                "Data_Source": "Yahoo Finance native weekly OHLCV",
                "Status": "OK",
            })
            rows.append(scored)
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "SCORING", "Error": repr(exc)})
            rows.append({
                "Symbol": symbol, "Liquidity_Rank": i,
                "Average_Dollar_Turnover_4W": float(item.Average_Dollar_Turnover_4W),
                "Status": "ERROR", "Error": repr(exc),
            })

    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking = ranking.sort_values(
            ["Final_Score", "EMA_Persistence_Score"],
            ascending=[False, False], na_position="last"
        ).reset_index(drop=True)
        ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    ranking.to_csv(out / "us_relative_vfi_early_accumulation_ranking.csv", index=False)
    liquidity_df.sort_values("Average_Dollar_Turnover_4W", ascending=False).to_csv(
        out / "us_liquidity_universe.csv", index=False
    )

    # Monthly Volume Divergence is calculated from monthly OHLCV bars derived
    # solely from the already-downloaded native Yahoo weekly candles. This is
    # explicitly an approximation; no daily data is fetched or resampled.
    monthly_rows = []
    for i, item in enumerate(selected.itertuples(index=False), 1):
        symbol = item.Symbol
        print(f"[MONTHLY DIVERGENCE {i}/{len(selected)}] {symbol}", flush=True)
        try:
            monthly = monthly_from_weekly(weekly_data[symbol])
            divergence = VolumeDivergenceMM().calculate(monthly)
            valid = pd.to_numeric(divergence["Volume_Divergence_MM"], errors="coerce").dropna()
            scored = score_monthly_divergence(valid)
            latest = divergence.loc[valid.index[-1]]
            scored.update({
                "Symbol": symbol,
                "Monthly_Candle_Date": valid.index[-1].strftime("%Y-%m-%d"),
                "Monthly_Bars": int(len(monthly)),
                "Monthly_Divergence_Bars": int(len(valid)),
                "Monthly_Bullish_Divergence": bool(latest["Volume_Div_Bullish"]),
                "Monthly_Bearish_Divergence": bool(latest["Volume_Div_Bearish"]),
                "Monthly_Divergence_Signal": str(latest["Volume_Div_Signal"]),
                "Average_Dollar_Turnover_4W": float(item.Average_Dollar_Turnover_4W),
                "Monthly_Source": "Calendar-month OHLCV aggregated from native Yahoo 1wk candles",
                "Status": "OK",
            })
            monthly_rows.append(scored)
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "MONTHLY_DIVERGENCE", "Error": repr(exc)})
            monthly_rows.append({"Symbol": symbol, "Status": "ERROR", "Error": repr(exc)})

    monthly_ranking = pd.DataFrame(monthly_rows)
    if not monthly_ranking.empty:
        monthly_ranking = monthly_ranking.sort_values(
            ["Monthly_Divergence_Final_Score"], ascending=False, na_position="last"
        ).reset_index(drop=True)
        monthly_ranking.insert(0, "Rank", range(1, len(monthly_ranking) + 1))
    monthly_ranking.to_csv(out / "us_monthly_volume_divergence_ranking.csv", index=False)

    weekly_valid = ranking[ranking.get("Status", pd.Series(index=ranking.index, dtype=str)).eq("OK")][
        ["Symbol", "Final_Score"]
    ].rename(columns={"Final_Score": "Weekly_VFI_Score"})
    monthly_valid = monthly_ranking[
        monthly_ranking.get("Status", pd.Series(index=monthly_ranking.index, dtype=str)).eq("OK")
    ][["Symbol", "Monthly_Divergence_Final_Score"]]
    combined = weekly_valid.merge(monthly_valid, on="Symbol", how="inner")
    if not combined.empty:
        combined["Combined_Score"] = (
            0.50 * combined["Weekly_VFI_Score"]
            + 0.50 * combined["Monthly_Divergence_Final_Score"]
        )
        combined = combined.sort_values("Combined_Score", ascending=False).reset_index(drop=True)
        combined.insert(0, "Rank", range(1, len(combined) + 1))
    combined.to_csv(out / "us_combined_weekly_vfi_monthly_divergence_ranking.csv", index=False)

    summary = {
        "requested_top": args.top,
        "seed_universe_size": len(symbols),
        "eligible_assets_with_native_weekly_data": int(len(eligible)),
        "ranked_assets": int((ranking.get("Status", pd.Series(dtype=str)) == "OK").sum()),
        "monthly_divergence_ranked_assets": int((monthly_ranking.get("Status", pd.Series(dtype=str)) == "OK").sum()),
        "combined_ranked_assets": int(len(combined)),
        "failed_assets": errors,
        "weekly_data_source": "Yahoo Finance chart history interval=1wk via yfinance; no daily interval requested and no daily-to-weekly resampling",
        "latest_completed_week": (
            str(ranking["Weekly_Candle_Date"].max()) if not ranking.empty else None
        ),
        "incomplete_current_week_excluded": True,
        "missing_week_policy": "Fill missing weekly slots with prior close as flat OHLC, volume=0, Synthetic_Bar=True; never invent volume",
        "liquidity_definition": "Mean Close*Volume over the latest four completed native Yahoo weekly candles",
        "minimum_history": "330 completed weekly bars and at least 200 valid VFI bars so VFI(130) and EMA200 are fully warmed",
        "ema_calculation": "EMA5/20/80/200 calculated over full valid weekly VFI history; percentiles use trailing 260 prior observations",
        "score_weights": {
            "EMA_persistence": 0.45, "EMA_signed_level": 0.30,
            "EMA_acceleration": 0.20, "early_accumulation_below_zero": 0.05
        },
        "ema_timeframe_weights": {"EMA5": 0.05, "EMA20": 0.15, "EMA80": 0.30, "EMA200": 0.50},
        "weekly_score_policy": "EMA-only: raw VFI is used only to construct EMA curves; VFI value, sign, near-zero proximity, raw VFI slopes, raw VFI acceleration and R2 do not affect score or penalties. Early accumulation rewards negative EMAs with positive sustained slopes; falling EMAs are penalized in proportion to their timeframe weight.",
        "ranking_scope": "Three outputs: weekly VFI, monthly Volume Divergence, and 50/50 combined ranking",
        "monthly_divergence_method": "Volume Divergence MM calculated on completed calendar-month OHLCV aggregated from native Yahoo 1wk candles; EMA5/20/80 weighted 10/30/60; EMA200 excluded. Monthly candles are approximations because a weekly candle crossing a calendar-month boundary cannot be split without daily data.",
        "combined_method": "50% weekly VFI Final_Score + 50% monthly Volume Divergence Final_Score; only assets with both valid scores are included.",
        "caveat": "Heuristic technical ranking for research, not a prediction or investment recommendation",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(ranking.head(30).to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
