"""Compare common volume-flow indicators on the 17 example assets.

The indicators are proxies for buying/selling pressure, not direct observation
of institutional orders. Uses completed daily and weekly Yahoo Finance candles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.data.market_data import MarketDataService

DEFAULT_SYMBOLS = [
    "DDD", "INTC", "NVDA", "BTC-USD", "ETH-USD", "LTC-USD",
    "EEM", "SPY", "AMD", "SQM", "EC", "AU", "CIB", "NU",
    "ISA.CL", "GEB.CL", "MINEROS.CL",
]


def completed_daily(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    idx = pd.to_datetime(data.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    result = data.copy()
    result.index = idx
    return result.loc[result.index.normalize() < pd.Timestamp.now().normalize()].copy()


def to_weekly(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    weekly = data.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    today = pd.Timestamp.now().normalize()
    last_friday = today - pd.Timedelta(days=(today.weekday() - 4) % 7)
    if today.weekday() == 4:
        last_friday -= pd.Timedelta(days=7)
    return weekly.loc[weekly.index.normalize() <= last_friday].copy()


def _score_slope(series: pd.Series, volume: pd.Series, window: int) -> float:
    """Convert recent flow-line slope to a comparable 0..100 score."""
    s = pd.to_numeric(series, errors="coerce").dropna().tail(window)
    v = pd.to_numeric(volume, errors="coerce").dropna().tail(window)
    if len(s) < max(6, window // 2) or v.empty:
        return float("nan")
    x = np.arange(len(s), dtype=float)
    slope = float(np.polyfit(x, s.to_numpy(dtype=float), 1)[0])
    typical_volume = float(v.replace([np.inf, -np.inf], np.nan).median())
    if not np.isfinite(typical_volume) or typical_volume <= 0:
        return float("nan")
    return float(np.clip(50.0 + 50.0 * np.tanh(4.0 * slope / typical_volume), 0, 100))


def calculate_indicators(data: pd.DataFrame, *, window: int = 20) -> dict:
    """Calculate OBV, A/D, CMF(20), MFI(14), relative volume and a composite."""
    required = ["High", "Low", "Close", "Volume"]
    if data.empty or any(c not in data.columns for c in required):
        raise ValueError("Need non-empty OHLCV data")
    df = data[required].apply(pd.to_numeric, errors="coerce").copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["Close"] > 0) & (df["Volume"] >= 0) & (df["High"] >= df["Low"])]
    if len(df) < 30:
        raise ValueError(f"Insufficient valid bars: {len(df)}")
    high, low, close, volume = (df[c] for c in required)
    prev_close = close.shift(1)

    # On-Balance Volume: add/subtract volume by close-to-close direction.
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()

    # Chaikin Accumulation/Distribution Line.
    spread = (high - low).replace(0, np.nan)
    multiplier = ((close - low) - (high - close)) / spread
    multiplier = multiplier.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1, 1)
    ad_line = (multiplier * volume).cumsum()

    # Chaikin Money Flow: recent money-flow volume divided by recent volume.
    mfv = multiplier * volume
    vol_sum = volume.rolling(window, min_periods=window).sum()
    cmf_volume_sum = volume.rolling(20, min_periods=20).sum()\n    cmf = mfv.rolling(20, min_periods=20).sum() / cmf_volume_sum.replace(0, np.nan)

    # Money Flow Index, volume-weighted momentum oscillator (0..100).
    typical = (high + low + close) / 3.0
    raw_money = typical * volume
    delta = typical.diff()
    positive = raw_money.where(delta > 0, 0.0)
    negative = raw_money.where(delta < 0, 0.0).abs()
    pos_sum = positive.rolling(14, min_periods=14).sum()
    neg_sum = negative.rolling(14, min_periods=14).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))
    mfi = mfi.mask((neg_sum == 0) & (pos_sum > 0), 100.0)
    mfi = mfi.mask((pos_sum == 0) & (neg_sum > 0), 0.0)

    obv_score = _score_slope(obv, volume, window)
    ad_score = _score_slope(ad_line, volume, window)
    cmf_value = float(cmf.dropna().iloc[-1]) if cmf.notna().any() else float("nan")
    mfi_value = float(mfi.dropna().iloc[-1]) if mfi.notna().any() else float("nan")
    cmf_score = float(np.clip(50 + 50 * cmf_value, 0, 100)) if np.isfinite(cmf_value) else float("nan")
    scores = {"OBV_Trend_Score": obv_score, "AD_Trend_Score": ad_score,
              "CMF_Score": cmf_score, "MFI_Score": mfi_value}
    valid_scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    # Reweight available indicators only; never treat missing history as zero.
    weights = {"OBV_Trend_Score": 0.30, "AD_Trend_Score": 0.25,
               "CMF_Score": 0.25, "MFI_Score": 0.20}
    total_weight = sum(weights[k] for k in valid_scores)
    composite = (sum(valid_scores[k] * weights[k] for k in valid_scores) / total_weight
                 if total_weight else float("nan"))
    last_volume = float(volume.iloc[-1])
    avg_volume = float(volume.rolling(20, min_periods=20).mean().iloc[-1])
    relative_volume = last_volume / avg_volume if avg_volume > 0 else float("nan")
    price_return_20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else float("nan")
    price_return_60 = float(close.iloc[-1] / close.iloc[-61] - 1.0) if len(close) >= 61 else float("nan")
    label = "ACCUMULATION_PRESSURE" if composite >= 60 else (
        "DISTRIBUTION_PRESSURE" if composite <= 40 else "MIXED / NEUTRAL"
    ) if np.isfinite(composite) else "INSUFFICIENT_DATA"
    # Potential stealth accumulation: positive flow score while price is flat/down.
    stealth = bool(np.isfinite(composite) and composite >= 60 and
                   np.isfinite(price_return_20) and price_return_20 <= 0.05 and
                   ((np.isfinite(obv_score) and obv_score > 50) or
                    (np.isfinite(ad_score) and ad_score > 50)))
    return {
        **scores,
        "Accumulation_Distribution_Score": float(composite),
        "Distribution_Pressure_Score": float(100.0 - composite) if np.isfinite(composite) else float("nan"),
        "Pressure_Classification": label,
        "Potential_Stealth_Accumulation": stealth,
        "OBV_Slope_Normalized": float(obv_score - 50) if np.isfinite(obv_score) else float("nan"),
        "AD_Slope_Normalized": float(ad_score - 50) if np.isfinite(ad_score) else float("nan"),
        "CMF_20": cmf_value,
        "MFI_14": mfi_value,
        "Relative_Volume_20D": relative_volume,
        "Price_Return_20B": price_return_20,
        "Price_Return_60B": price_return_60,
        "Indicator_Count": len(valid_scores),
        "Bars_Used": int(len(df)),
    }


def analyze_symbol(symbol: str, daily: pd.DataFrame) -> dict:
    base = {"Symbol": symbol, "Analysis_Status": "ERROR"}
    try:
        daily = completed_daily(daily)
        if daily.empty:
            raise ValueError("No completed daily candles")
        d = calculate_indicators(daily)
        try:
            weekly = to_weekly(daily)
            w = calculate_indicators(weekly, window=13)
            w = {f"Weekly_{k}": v for k, v in w.items()
                 if k in ("Accumulation_Distribution_Score", "Pressure_Classification",
                          "OBV_Trend_Score", "AD_Trend_Score", "CMF_Score", "MFI_Score",
                          "CMF_20", "MFI_14", "Bars_Used")}
        except Exception as exc:
            w = {"Weekly_Analysis_Status": f"UNAVAILABLE: {exc}"}
        base.update(d)
        base.update(w)
        base.update({
            "Analysis_Status": "ANALYZED",
            "Daily_Candle_Date": daily.index[-1].strftime("%Y-%m-%d"),
            "Weekly_Candle_Date": to_weekly(daily).index[-1].strftime("%Y-%m-%d") if not to_weekly(daily).empty else "",
            "Price": float(daily["Close"].iloc[-1]),
        })
    except Exception as exc:
        base["Error"] = repr(exc)
    return base


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Score accumulation/distribution using OBV, A/D, CMF and MFI.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--period", default="max")
    parser.add_argument("--output-dir", default="reports/volume_accumulation")
    args = parser.parse_args()
    symbols = list(dict.fromkeys(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    if not symbols:
        parser.error("Provide at least one symbol")
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    service = MarketDataService()
    rows = []
    for i, symbol in enumerate(symbols, start=1):
        print(f"[{i}/{len(symbols)}] {symbol}", flush=True)
        try:
            raw = service.get_daily_data(symbol, period=args.period)
            row = analyze_symbol(symbol, raw)
            meta = service.get_last_metadata()
            row["Data_Provider"] = meta.get("Provider", "Yahoo Finance / yfinance")
            row["Data_First_Date"] = meta.get("First_Date")
            row["Data_Last_Date"] = meta.get("Last_Date")
        except Exception as exc:
            row = {"Symbol": symbol, "Analysis_Status": "ERROR", "Error": repr(exc)}
        rows.append(row)
    result = pd.DataFrame(rows)
    if "Accumulation_Distribution_Score" in result:
        result = result.sort_values(
            ["Accumulation_Distribution_Score", "Symbol"],
            ascending=[False, True], na_position="last"
        ).reset_index(drop=True)
        result.insert(0, "Rank", range(1, len(result) + 1))
    result.to_csv(out / "all_example_assets_volume_indicators.csv", index=False)
    result[result.get("Pressure_Classification", pd.Series(index=result.index, dtype=str)) == "ACCUMULATION_PRESSURE"].to_csv(
        out / "accumulation_pressure_candidates.csv", index=False
    )
    result[result.get("Pressure_Classification", pd.Series(index=result.index, dtype=str)) == "DISTRIBUTION_PRESSURE"].to_csv(
        out / "distribution_pressure_candidates.csv", index=False
    )
    summary = {
        "symbols_requested": symbols,
        "assets_reported": int(len(result)),
        "analysis_status_counts": result.get("Analysis_Status", pd.Series(dtype=str)).value_counts().to_dict(),
        "daily_indicators": ["OBV", "Accumulation/Distribution Line", "Chaikin Money Flow (20)", "Money Flow Index (14)"],
        "weekly_confirmation": "Same indicators recalculated on completed weekly candles when history permits",
        "composite_weights": {"OBV trend": 0.30, "A/D trend": 0.25, "CMF": 0.25, "MFI": 0.20},
        "score_interpretation": ">=60 accumulation pressure; <=40 distribution pressure; otherwise mixed. This is a technical proxy, not a direct measurement of institutional orders.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    cols = ["Rank", "Symbol", "Price", "Accumulation_Distribution_Score", "Distribution_Pressure_Score",
            "Pressure_Classification", "Potential_Stealth_Accumulation", "OBV_Trend_Score",
            "AD_Trend_Score", "CMF_20", "MFI_14", "Relative_Volume_20D",
            "Weekly_Accumulation_Distribution_Score", "Weekly_Pressure_Classification", "Price_Return_20B"]
    print(json.dumps(summary, indent=2, default=str))
    print(result[[c for c in cols if c in result.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
