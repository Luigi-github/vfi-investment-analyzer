"""Build monthly Volume Divergence MM and combined rankings for the exact weekly top-200 universe.

Reads the OHLCV cache and weekly screen generated earlier in the same workflow.
All EMAs are calculated on the indicator series, never on price.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.indicators.volume_divergence_mm import VolumeDivergenceMM
REPORTS = ROOT / "reports/us_weekly_then_daily"
CACHE = ROOT / "data/market_cache/us_equities"


def cache_path(symbol: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in symbol)
    return CACHE / f"{safe}.csv"


def completed_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.copy().sort_index()
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    # Only completed calendar months; never use the partial current month.
    this_month = pd.Timestamp.now().normalize().replace(day=1)
    data = data[data.index < this_month]
    if data.empty:
        return data
    monthly = data.resample("ME").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return monthly


def slope(series: pd.Series, bars: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().tail(bars + 1)
    if len(s) < bars + 1:
        return float("nan")
    return float((s.iloc[-1] - s.iloc[0]) / bars)


def score_monthly(symbol: str, daily: pd.DataFrame) -> dict:
    monthly = completed_monthly(daily)
    if len(monthly) < 77:
        raise ValueError(f"insufficient completed monthly candles: {len(monthly)}; need >=77 for the indicator warm-up")
    frame = VolumeDivergenceMM(
        ema_short=5, ema_medium=20, ema_long=80
    ).calculate(monthly)
    s = pd.to_numeric(frame["Volume_Divergence_MM"], errors="coerce").dropna()
    if len(s) < 1:
        raise ValueError("no valid monthly divergence observations after the WMA warm-up")
    s = s.tail(120)
    e5 = s.ewm(span=5, adjust=False, min_periods=min(5, len(s))).mean()
    e20 = s.ewm(span=20, adjust=False, min_periods=min(20, len(s))).mean()
    e80 = s.ewm(span=80, adjust=False, min_periods=min(80, len(s))).mean()
    slope1, slope3, slope6 = slope(s, 1), slope(s, 3), slope(s, 6)
    rising_negative = (s < 0) & (slope_series(s, 1) > 0) & (slope_series(s, 3) > 0)
    falling_positive = (s > 0) & (slope_series(s, 1) < 0) & (slope_series(s, 3) < 0)
    positive_points = int(rising_negative.sum())
    distribution_points = int(falling_positive.sum())
    last = float(s.iloc[-1])
    e5v, e20v, e80v = float(e5.iloc[-1]), float(e20.iloc[-1]), float(e80.iloc[-1])
    e5r, e20r, e80r = slope(e5, 3), slope(e20, 3), slope(e80, 3)
    weights = {"ema5": 0.10, "ema20": 0.30, "ema80": 0.60}
    # Rewards the bullish EMA ordering while the oscillator remains below zero.
    structure = (
        weights["ema5"] * float(e5v > e20v and e5r > 0 and e5v < 0)
        + weights["ema20"] * float(e20v > e80v and e20r > 0 and e20v < 0)
        + weights["ema80"] * float(e80r > 0 and e80v < 0)
    )
    inversion = (
        weights["ema5"] * float(e5v < e20v)
        + weights["ema20"] * float(e20v < e80v)
        + weights["ema80"] * float(e80v >= 0)
    )
    hist_median = float(s.median())
    hist_q95 = float(s.quantile(0.95))
    overextension = (
        float(np.clip((last - hist_median) / (hist_q95 - hist_median), 0, 1))
        if hist_q95 > hist_median and last > hist_median else 0.0
    )
    base = positive_points - distribution_points
    scale_points = max(positive_points, 1)
    score = (
        base
        + 0.75 * scale_points * structure
        - 0.75 * scale_points * inversion
        - 0.50 * scale_points * overextension
        - 0.10 * scale_points * float(slope3 < 0)
        - 0.10 * scale_points * float(slope6 < 0)
        - 0.25 * scale_points * float(e5v > 0 and e20v > 0 and e80v > 0)
    )
    return {
        "Symbol": symbol,
        "Monthly_Score": float(score),
        "Monthly_Divergence": last,
        "Monthly_EMA5": e5v, "Monthly_EMA20": e20v, "Monthly_EMA80": e80v,
        "Monthly_EMA5_Slope_3M": e5r, "Monthly_EMA20_Slope_3M": e20r,
        "Monthly_EMA80_Slope_3M": e80r,
        "Monthly_Slope_1M": slope1, "Monthly_Slope_3M": slope3, "Monthly_Slope_6M": slope6,
        "Months_Negative_Rising": positive_points,
        "Months_Positive_Falling": distribution_points,
        "EMA_Structure_Bonus": 0.75 * scale_points * structure,
        "EMA_Inversion_Penalty": 0.75 * scale_points * inversion,
        "Overextension_Penalty": 0.50 * scale_points * overextension,
        "Monthly_Candle_Date": monthly.index[-1].strftime("%Y-%m-%d"),
        "Monthly_Bars": int(len(monthly)),
        "Valid_Monthly_Divergence_Bars": int(len(pd.to_numeric(frame["Volume_Divergence_MM"], errors="coerce").dropna())),
        "EMA80_Warmup_Complete": bool(len(s) >= 80),
    }


def slope_series(series: pd.Series, bars: int) -> pd.Series:
    # Difference-per-bar slope, causal at each monthly observation.
    return (series - series.shift(bars)) / bars


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    weekly_path = REPORTS / "us_top_200_weekly_screen.csv"
    if not weekly_path.exists():
        raise FileNotFoundError(f"weekly ranking not found: {weekly_path}")
    weekly = pd.read_csv(weekly_path)
    if "Symbol" not in weekly or "Weekly_Opportunity_Score" not in weekly:
        raise ValueError("weekly screen lacks Symbol or Weekly_Opportunity_Score")
    symbols = weekly["Symbol"].dropna().astype(str).drop_duplicates().tolist()
    if len(symbols) != 200:
        raise RuntimeError(f"expected the same 200-symbol weekly universe, found {len(symbols)}")
    monthly_rows, errors = [], []
    for i, symbol in enumerate(symbols, 1):
        print(f"[MONTHLY DIVERGENCE {i}/{len(symbols)}] {symbol}", flush=True)
        try:
            path = cache_path(symbol)
            if not path.exists():
                raise FileNotFoundError(f"OHLCV cache missing: {path.name}")
            daily = pd.read_csv(path, index_col=0, parse_dates=True)
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(daily.columns):
                raise ValueError(f"missing OHLCV columns: {sorted(required - set(daily.columns))}")
            row = score_monthly(symbol, daily)
            monthly_rows.append(row)
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": repr(exc)})
            monthly_rows.append({"Symbol": symbol, "Monthly_Score": np.nan, "Monthly_Status": f"ERROR: {exc!r}"})
    monthly = pd.DataFrame(monthly_rows)
    monthly["Monthly_Rank"] = monthly["Monthly_Score"].rank(method="min", ascending=False, na_option="bottom").astype(int)
    monthly = monthly.sort_values(["Monthly_Score", "Symbol"], ascending=[False, True], na_position="last")
    monthly.to_csv(REPORTS / "us_top_200_monthly_divergence_ranking.csv", index=False)

    # Percentile ranks make the combined score comparable even though raw VFI
    # and Volume Divergence MM use different numerical scales.
    combined = weekly[["Symbol", "Opportunity_Rank", "Weekly_Opportunity_Score", "Weekly_VFI",
                       "Weekly_Candle_Date"]].copy()
    combined = combined.merge(monthly[["Symbol", "Monthly_Rank", "Monthly_Score",
                                       "Monthly_Divergence", "Monthly_Candle_Date"]],
                              on="Symbol", how="left", validate="one_to_one")
    n = len(combined)
    combined["Weekly_Percentile"] = 100.0 * (n - combined["Opportunity_Rank"]) / max(n - 1, 1)
    valid_monthly = combined["Monthly_Score"].notna()
    combined["Monthly_Percentile"] = np.where(
        valid_monthly, 100.0 * (n - combined["Monthly_Rank"]) / max(n - 1, 1), 0.0
    )
    combined["Combined_Score"] = 0.5 * combined["Weekly_Percentile"] + 0.5 * combined["Monthly_Percentile"]
    combined["Combined_Rank"] = combined["Combined_Score"].rank(method="min", ascending=False).astype(int)
    combined = combined.sort_values(["Combined_Score", "Weekly_Opportunity_Score"], ascending=False)
    combined.to_csv(REPORTS / "us_top_200_combined_ranking.csv", index=False)
    (REPORTS / "three_rankings_summary.json").write_text(json.dumps({
        "weekly_assets": int(len(weekly)),
        "monthly_assets_scored": int(monthly["Monthly_Score"].notna().sum()),
        "monthly_errors": errors,
        "weekly_latest_completed_candle": str(weekly.get("Weekly_Candle_Date", pd.Series(dtype=str)).dropna().max()),
        "monthly_latest_completed_candle": str(monthly.get("Monthly_Candle_Date", pd.Series(dtype=str)).dropna().max()),
        "ema_weights": {"weekly_vfi": {"EMA5": 0.05, "EMA20": 0.15, "EMA80": 0.30, "EMA200": 0.50},
                        "monthly_divergence": {"EMA5": 0.10, "EMA20": 0.30, "EMA80": 0.60}},
        "combined_method": "equal-weight mean of within-universe weekly and monthly rank percentiles; 0-100 scale",
        "same_universe": True,
    }, indent=2), encoding="utf-8")
    print("\nTOP 20 MONTHLY DIVERGENCE", flush=True)
    print(monthly.head(20).to_string(index=False), flush=True)
    print("\nTOP 20 COMBINED", flush=True)
    print(combined.head(20).to_string(index=False), flush=True)
    print(f"\nSaved weekly, monthly and combined ranking reports to {REPORTS}", flush=True)


if __name__ == "__main__":
    main()
