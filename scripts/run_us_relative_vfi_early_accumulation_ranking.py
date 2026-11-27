"""Relative, weekly VFI early-accumulation ranking.

All EMA curves are calculated on the VFI series (never on price). Historical
percentiles are asset-relative and use only observations available up to the
current completed weekly candle. This is a research heuristic, not a forecast.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.indicators.vfi import VFIIndicator
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
    """Score a completed weekly VFI series on comparable, asset-relative scales."""
    s = pd.to_numeric(vfi, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 220:
        raise ValueError(f"Insufficient valid weekly VFI history ({len(s)}; need >=220)")

    # Critical: calculate each EMA on the FULL series, then use trailing history
    # only for percentile estimates. Slicing before EMA would truncate EMA200.
    emas = {
        key: s.ewm(span=period, adjust=False, min_periods=period).mean()
        for key, period in EMA_PERIODS.items()
    }
    ema_now = {key: float(series.iloc[-1]) for key, series in emas.items()}
    ema_streaks = {
        key: trailing_true_streak(series.diff() > 0)
        for key, series in emas.items()
    }
    ema_streak_score = sum(
        100.0 * EMA_WEIGHTS[key] * min(ema_streaks[key], STREAK_CAP) / STREAK_CAP
        for key in EMA_PERIODS
    )

    ema_percentiles = {
        key: historical_percentile(series.iloc[:-1], ema_now[key])
        for key, series in emas.items()
    }
    # Signed EMA level score: negative EMA values add points; positive values
    # subtract points in proportion to their own historical percentile. Thus a
    # mildly positive EMA at a low percentile is penalized lightly, while an
    # unusually positive EMA at a high percentile is penalized more strongly.
    # The weighted signed result is mapped to [0, 100] for a stable score range.
    ema_signed_components = {}
    for key in EMA_PERIODS:
        pct = ema_percentiles[key]
        value = ema_now[key]
        if not np.isfinite(pct) or not np.isfinite(value):
            ema_signed_components[key] = float("nan")
        elif value < 0:
            ema_signed_components[key] = 100.0 - pct
        elif value > 0:
            ema_signed_components[key] = -pct
        else:
            ema_signed_components[key] = 0.0
    available_ema_weights = sum(
        EMA_WEIGHTS[key] for key in EMA_PERIODS
        if np.isfinite(ema_signed_components[key])
    )
    ema_signed_raw_score = (
        sum(EMA_WEIGHTS[key] * ema_signed_components[key]
            for key in EMA_PERIODS if np.isfinite(ema_signed_components[key]))
        / available_ema_weights
        if available_ema_weights else 0.0
    )
    ema_level_score = float(np.clip(50.0 + 0.5 * ema_signed_raw_score, 0.0, 100.0))

    current = float(s.iloc[-1])
    vfi_pct = historical_percentile(s.iloc[:-1], current)
    abs_history = s.iloc[:-1].abs()
    abs_current_pct = historical_percentile(abs_history, abs(current))
    # Position near zero is measured relative to this asset's own historical
    # absolute VFI distribution, not the raw VFI magnitude.
    near_zero_score = float(np.clip(100.0 - abs_current_pct, 0.0, 100.0))

    # VFI persistence: current uninterrupted rise, recent up-week ratio, and
    # linearity, with R² rewarded only if the recent slope is positive.
    vfi_streak = trailing_true_streak(s.diff() > 0)
    streak_component = min(vfi_streak, 13) / 13.0 * 100.0
    recent13 = s.tail(13)
    up_ratio = float((recent13.diff().dropna() > 0).mean()) if len(recent13) > 1 else 0.0
    slope4 = regression_slope(s.tail(4).to_numpy())
    prior4 = regression_slope(s.iloc[-8:-4].to_numpy()) if len(s) >= 8 else float("nan")
    slope13 = regression_slope(recent13.to_numpy())
    r2_13 = linear_r2(recent13.to_numpy())
    quality_component = r2_13 * 100.0 if np.isfinite(slope13) and slope13 > 0 else r2_13 * 35.0
    vfi_persistence_score = 0.45 * streak_component + 0.35 * up_ratio * 100.0 + 0.20 * quality_component

    # Acceleration = recent 4-week slope minus the preceding 4-week slope.
    # Compare it to the asset's own historical acceleration distribution.
    slope4_series = s.rolling(4).apply(regression_slope, raw=True)
    acceleration = slope4_series.diff(4)
    acceleration_pct = historical_percentile(acceleration.iloc[:-1], float(acceleration.iloc[-1]))
    slope_pct = historical_percentile(s.diff(), float(s.diff().iloc[-1]))
    if not np.isfinite(acceleration_pct):
        acceleration_pct = 50.0
    if not np.isfinite(slope_pct):
        slope_pct = 50.0
    acceleration_score = 0.70 * acceleration_pct + 0.30 * slope_pct
    # A decelerating decline is an early watch signal, but less attractive than
    # positive momentum already confirmed.
    if not (np.isfinite(slope4) and slope4 > 0):
        acceleration_score *= 0.60
    if np.isfinite(slope4) and np.isfinite(prior4) and slope4 > 0 and prior4 <= 0:
        acceleration_score = min(100.0, acceleration_score + 8.0)

    # Early-accumulation blocks. Relative proximity to zero is asset-normalized
    # by the historical percentile of |VFI|. It receives 5% of the total score;
    # EMA signed level receives 20%, preserving the 25% position block.
    persistence_block = ema_streak_score
    near_zero_block = near_zero_score
    relative_position_block = 0.80 * ema_level_score + 0.20 * near_zero_block
    trend_block = float(np.clip(vfi_persistence_score, 0.0, 100.0))
    acceleration_block = float(np.clip(acceleration_score, 0.0, 100.0))
    gross_score = (
        0.30 * persistence_block
        + 0.20 * ema_level_score
        + 0.05 * near_zero_block
        + 0.25 * trend_block
        + 0.20 * acceleration_block
    )

    # Penalize deterioration, but distinguish a pause from a sustained pullback.
    slope8 = regression_slope(s.tail(8).to_numpy())
    slope26 = regression_slope(s.tail(26).to_numpy())
    ema_negative_weights = sum(
        EMA_WEIGHTS[key] for key, series in emas.items()
        if len(series.dropna()) >= 2 and float(series.diff().iloc[-1]) < 0
    )
    pullback_penalty = 0.0
    if np.isfinite(slope4) and slope4 < 0:
        pullback_penalty += 0.06 * gross_score
    if np.isfinite(slope8) and slope8 < 0:
        pullback_penalty += 0.05 * gross_score
    if np.isfinite(slope13) and slope13 < 0:
        pullback_penalty += 0.07 * gross_score
    if np.isfinite(slope26) and slope26 < 0:
        pullback_penalty += 0.04 * gross_score
    # EMA deterioration adds a smaller confirmation penalty, not a single-week cliff.
    ema_pullback_penalty = 0.10 * gross_score * ema_negative_weights
    # A short pause (flat slope) is not treated like a falling trend.
    if np.isfinite(slope4) and slope4 >= 0 and np.isfinite(slope13) and slope13 >= 0:
        pullback_penalty *= 0.50

    positive_vfi_penalty = 0.0
    if current > 0 and np.isfinite(vfi_pct):
        positive_vfi_penalty = 0.25 * max(0.0, vfi_pct - 50.0)
    positive_ema_weights = sum(
        EMA_WEIGHTS[key] for key, value in ema_now.items() if value > 0
    )
    all_positive_penalty = 8.0 if positive_ema_weights >= 0.999 else 0.0
    ema_structure_penalty = 0.0
    if ema_now["ema5"] < ema_now["ema20"]:
        ema_structure_penalty += 2.0
    if ema_now["ema20"] < ema_now["ema80"]:
        ema_structure_penalty += 4.0
    if ema_now["ema80"] < ema_now["ema200"]:
        ema_structure_penalty += 6.0

    total_penalty = min(
        positive_vfi_penalty + all_positive_penalty + ema_structure_penalty
        + pullback_penalty + ema_pullback_penalty,
        0.55 * gross_score,
    )
    final_score = float(np.clip(gross_score - total_penalty, 0.0, 100.0))

    return {
        "VFI": current,
        "VFI_Historical_Percentile": vfi_pct,
        "VFI_Absolute_Magnitude_Percentile": abs_current_pct,
        "VFI_Near_Zero_Score": near_zero_score,
        "VFI_Near_Zero_Contribution": 0.05 * near_zero_score,
        "EMA_Level_Contribution": 0.20 * ema_level_score,
        "VFI_Up_Streak_Weeks": vfi_streak,
        "VFI_Up_Weeks_Ratio_13W": up_ratio,
        "VFI_Slope_4W": slope4,
        "VFI_Slope_Previous_4W": prior4,
        "VFI_Slope_13W": slope13,
        "VFI_Slope_8W": slope8,
        "VFI_Slope_26W": slope26,
        "VFI_Acceleration_4W": float(acceleration.iloc[-1]) if len(acceleration) else float("nan"),
        "VFI_Acceleration_Percentile": acceleration_pct,
        "VFI_Persistence_Score": trend_block,
        "EMA_Persistence_Score": persistence_block,
        "EMA_Level_Signed_Score": ema_level_score,
        "EMA_Level_Signed_Raw_Score": ema_signed_raw_score,
        **{f"{key.upper()}_Signed_Level_Component": value for key, value in ema_signed_components.items()},
        "Relative_Position_Score": relative_position_block,
        "Acceleration_Score": acceleration_block,
        "Gross_Score": gross_score,
        "VFI_Positive_Extension_Penalty": positive_vfi_penalty,
        "All_EMAs_Positive_Penalty": all_positive_penalty,
        "EMA_Inversion_Penalty": ema_structure_penalty,
        "Pullback_Penalty": pullback_penalty,
        "EMA_Pullback_Penalty": ema_pullback_penalty,
        "EMA_Negative_Slope_Weight": ema_negative_weights,
        "Total_Penalty": total_penalty,
        "Final_Score": final_score,
        **{f"{key.upper()}_Value": value for key, value in ema_now.items()},
        **{f"{key.upper()}_Percentile": value for key, value in ema_percentiles.items()},
        **{f"{key.upper()}_Rising_Streak_Weeks": value for key, value in ema_streaks.items()},
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

    for i, symbol in enumerate(symbols, 1):
        print(f"[WEEKLY YAHOO {i}/{len(symbols)}] {symbol}", flush=True)
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
            weekly_data[symbol] = weekly
            liquidity_rows.append({
                "Symbol": symbol,
                "Average_Dollar_Turnover_4W": dollar_turnover,
                "Data_Date": meta["Weekly_Candle_Date"],
                "Weekly_Bars": meta["Weekly_Bars"],
                "Synthetic_Weekly_Bars": meta["Synthetic_Weekly_Bars"],
                "Yahoo_Interval": "1wk",
                "Data_Source": meta["Source"],
                "Status": "OK",
            })
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "WEEKLY_DATA", "Error": repr(exc)})
            liquidity_rows.append({
                "Symbol": symbol, "Average_Dollar_Turnover_4W": np.nan,
                "Data_Date": "", "Weekly_Bars": 0, "Synthetic_Weekly_Bars": 0,
                "Yahoo_Interval": "1wk", "Data_Source": "Yahoo Finance native weekly OHLCV",
                "Status": f"ERROR: {exc!r}",
            })

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
    summary = {
        "requested_top": args.top,
        "seed_universe_size": len(symbols),
        "eligible_assets_with_native_weekly_data": int(len(eligible)),
        "ranked_assets": int((ranking.get("Status", pd.Series(dtype=str)) == "OK").sum()),
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
            "EMA_persistence": 0.30, "EMA_signed_level": 0.20,
            "VFI_relative_near_zero": 0.05,
            "VFI_persistence_and_quality": 0.25, "VFI_acceleration": 0.20
        },
        "ranking_scope": "VFI weekly ranking only; this workflow does not calculate monthly Volume Divergence or combined ranking",
        "caveat": "Heuristic technical ranking for research, not a prediction or investment recommendation",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(ranking.head(30).to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
