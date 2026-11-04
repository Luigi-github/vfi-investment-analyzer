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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from scripts.run_us_weekly_then_daily_screen import (
    US_LIQUID_UNIVERSE,
    completed_daily,
    load_or_download_daily,
    to_weekly,
)

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
    # Lower EMA percentiles are preferred, but cannot override trend/persistence.
    ema_low_score = sum(
        EMA_WEIGHTS[key] * (100.0 - ema_percentiles[key])
        for key in EMA_PERIODS if np.isfinite(ema_percentiles[key])
    ) / sum(EMA_WEIGHTS[key] for key in EMA_PERIODS if np.isfinite(ema_percentiles[key]))

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

    # Four normalized blocks. Each block has a common 0-100 scale.
    persistence_block = ema_streak_score
    relative_position_block = 0.60 * ema_low_score + 0.40 * near_zero_score
    trend_block = float(np.clip(vfi_persistence_score, 0.0, 100.0))
    acceleration_block = float(np.clip(acceleration_score, 0.0, 100.0))
    gross_score = (
        0.35 * persistence_block
        + 0.30 * relative_position_block
        + 0.20 * trend_block
        + 0.15 * acceleration_block
    )

    # Progressive overextension penalty: a positive VFI is not inherently bad;
    # it is penalized increasingly when high relative to its own history.
    positive_vfi_penalty = 0.0
    if current > 0 and np.isfinite(vfi_pct):
        positive_vfi_penalty = 0.25 * max(0.0, vfi_pct - 50.0)  # max 12.5 points

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
    negative_slope_penalty = 0.0
    if not np.isfinite(slope4) or slope4 < 0:
        negative_slope_penalty += 5.0
    if not np.isfinite(slope13) or slope13 < 0:
        negative_slope_penalty += 5.0

    total_penalty = min(
        positive_vfi_penalty + all_positive_penalty + ema_structure_penalty + negative_slope_penalty,
        0.30 * gross_score,
    )
    final_score = float(np.clip(gross_score - total_penalty, 0.0, 100.0))

    return {
        "VFI": current,
        "VFI_Historical_Percentile": vfi_pct,
        "VFI_Absolute_Magnitude_Percentile": abs_current_pct,
        "VFI_Near_Zero_Score": near_zero_score,
        "VFI_Up_Streak_Weeks": vfi_streak,
        "VFI_Up_Weeks_Ratio_13W": up_ratio,
        "VFI_Slope_4W": slope4,
        "VFI_Slope_Previous_4W": prior4,
        "VFI_Slope_13W": slope13,
        "VFI_Acceleration_4W": float(acceleration.iloc[-1]) if len(acceleration) else float("nan"),
        "VFI_Acceleration_Percentile": acceleration_pct,
        "VFI_Persistence_Score": trend_block,
        "EMA_Persistence_Score": persistence_block,
        "EMA_Low_Percentile_Score": ema_low_score,
        "Relative_Position_Score": relative_position_block,
        "Acceleration_Score": acceleration_block,
        "Gross_Score": gross_score,
        "VFI_Positive_Extension_Penalty": positive_vfi_penalty,
        "All_EMAs_Positive_Penalty": all_positive_penalty,
        "EMA_Inversion_Penalty": ema_structure_penalty,
        "Negative_Slope_Penalty": negative_slope_penalty,
        "Total_Penalty": total_penalty,
        "Final_Score": final_score,
        **{f"{key.upper()}_Value": value for key, value in ema_now.items()},
        **{f"{key.upper()}_Percentile": value for key, value in ema_percentiles.items()},
        **{f"{key.upper()}_Rising_Streak_Weeks": value for key, value in ema_streaks.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank US stocks by relative early accumulation in weekly VFI.")
    parser.add_argument("--symbols", default=",".join(US_LIQUID_UNIVERSE))
    parser.add_argument("--period", default="max")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--output-dir", default="reports/us_relative_vfi_ranking")
    parser.add_argument("--cache-dir", default="data/market_cache/us_equities")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    symbols = list(dict.fromkeys(x.strip().upper() for x in args.symbols.split(",") if x.strip()))
    service = MarketDataService()
    cache = ROOT / args.cache_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_data: dict[str, pd.DataFrame] = {}
    liquidity_rows, errors = [], []

    for i, symbol in enumerate(symbols, 1):
        print(f"[DATA {i}/{len(symbols)}] {symbol}", flush=True)
        try:
            data, source = load_or_download_daily(
                service, symbol, args.period, cache, refresh=args.refresh_cache
            )
            data = completed_daily(data)
            if data.empty or not {"Close", "Volume"}.issubset(data.columns):
                raise ValueError("No usable OHLCV data")
            raw_data[symbol] = data
            turnover = (pd.to_numeric(data["Close"], errors="coerce") *
                        pd.to_numeric(data["Volume"], errors="coerce")).replace([np.inf, -np.inf], np.nan).dropna().tail(20)
            liquidity = float(turnover.mean()) if len(turnover) >= 10 else float("nan")
            liquidity_rows.append({
                "Symbol": symbol, "Average_Dollar_Turnover_20D": liquidity,
                "Data_Date": data.index[-1].strftime("%Y-%m-%d"), "Data_Source": source,
            })
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "DATA", "Error": repr(exc)})
            liquidity_rows.append({"Symbol": symbol, "Average_Dollar_Turnover_20D": np.nan,
                                   "Data_Date": "", "Data_Source": f"ERROR: {exc!r}"})

    liquidity_df = pd.DataFrame(liquidity_rows)
    eligible = liquidity_df[np.isfinite(pd.to_numeric(liquidity_df["Average_Dollar_Turnover_20D"], errors="coerce"))]
    selected = eligible.sort_values("Average_Dollar_Turnover_20D", ascending=False).head(args.top)
    rows = []
    for i, item in enumerate(selected.itertuples(index=False), 1):
        symbol = item.Symbol
        print(f"[SCORE {i}/{len(selected)}] {symbol}", flush=True)
        try:
            weekly = to_weekly(raw_data[symbol])
            weekly = completed_daily(weekly)
            vfi_frame = VFIIndicator().calculate(weekly)
            vfi = pd.to_numeric(vfi_frame["VFI"], errors="coerce").dropna()
            scored = score_weekly_vfi(vfi)
            scored.update({
                "Symbol": symbol,
                "Liquidity_Rank": i,
                "Average_Dollar_Turnover_20D": float(item.Average_Dollar_Turnover_20D),
                "Weekly_Candle_Date": vfi.index[-1].strftime("%Y-%m-%d"),
                "Weekly_Bars": len(weekly),
                "Data_Date": item.Data_Date,
                "Status": "OK",
            })
            rows.append(scored)
        except Exception as exc:
            errors.append({"Symbol": symbol, "Stage": "SCORING", "Error": repr(exc)})
            rows.append({"Symbol": symbol, "Liquidity_Rank": i,
                         "Average_Dollar_Turnover_20D": float(item.Average_Dollar_Turnover_20D),
                         "Status": "ERROR", "Error": repr(exc)})

    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking = ranking.sort_values(["Final_Score", "EMA_Persistence_Score"], ascending=[False, False], na_position="last").reset_index(drop=True)
        ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    ranking.to_csv(out / "us_relative_vfi_early_accumulation_ranking.csv", index=False)
    liquidity_df.sort_values("Average_Dollar_Turnover_20D", ascending=False).to_csv(out / "us_liquidity_universe.csv", index=False)
    summary = {
        "requested_top": args.top,
        "universe_size": len(symbols),
        "ranked_assets": int((ranking.get("Status", pd.Series(dtype=str)) == "OK").sum()),
        "failed_assets": errors,
        "completed_weekly_candles_only": True,
        "ema_calculation": "EMA5/20/80/200 calculated over full valid weekly VFI history; percentiles use trailing 260 prior observations",
        "score_weights": {"EMA_persistence": 0.35, "relative_EMA_position_and_VFI_near_zero": 0.30,
                          "VFI_persistence_and_quality": 0.20, "VFI_acceleration": 0.15},
        "normalization": "Each EMA and VFI feature compared with its own trailing historical distribution; no raw cross-asset VFI magnitude used in ranking",
        "caveat": "Heuristic technical ranking for research, not a prediction or investment recommendation",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(ranking.head(30).to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
