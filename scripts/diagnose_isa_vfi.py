from __future__ import annotations

"""Audit ISA VFI against Yahoo daily-resampled and native weekly candles."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yfinance as yf

from app.indicators.vfi import VFIIndicator

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
LAST_COMPLETED = pd.Timestamp("2026-11-13")


def normalize_index(data: pd.DataFrame) -> pd.DataFrame:
    data = data[COLUMNS].copy()
    idx = pd.to_datetime(data.index, errors="coerce")
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    data.index = idx.normalize()
    return data.loc[~data.index.isna()].sort_index()


def daily_to_weekly(data: pd.DataFrame) -> pd.DataFrame:
    data = normalize_index(data)
    return data.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open", "High", "Low", "Close"]).loc[:LAST_COMPLETED]


def native_weekly(data: pd.DataFrame) -> pd.DataFrame:
    data = normalize_index(data)
    return data.loc[data.index <= LAST_COMPLETED].dropna(
        subset=["Open", "High", "Low", "Close"]
    )


def summarize(label: str, bars: pd.DataFrame) -> dict:
    calc = VFIIndicator(
        length=130, coef=0.2, vcoef=2.5, signal_length=5
    ).calculate(bars)
    latest = calc.dropna(subset=["VFI"]).iloc[-1]
    tail = calc.tail(8)
    return {
        "source": label,
        "weekly_rows": int(len(bars)),
        "first_week": bars.index.min().date().isoformat(),
        "last_week": bars.index.max().date().isoformat(),
        "latest_ohlcv": {k.lower(): float(latest[k]) for k in COLUMNS},
        "latest_components": {
            k: (float(latest[k]) if pd.notna(latest[k]) else None)
            for k in [
                "TypicalPrice", "Volatility", "Cutoff", "VolumeAverage",
                "VolumeMax", "CappedVolume", "MoneyFlow", "PriceMovement",
                "VolumeFlow", "RollingVolumeFlow",
            ]
        },
        "vfi": float(latest["VFI"]),
        "signal_5": float(latest["VFI_Signal_5"]),
        "last_8_bars": [
            {
                "date": ix.date().isoformat(),
                **{k.lower(): float(row[k]) for k in COLUMNS},
                "vfi": None if pd.isna(row["VFI"]) else float(row["VFI"]),
                "signal_5": None if pd.isna(row["VFI_Signal_5"]) else float(row["VFI_Signal_5"]),
            }
            for ix, row in tail.iterrows()
        ],
    }


def main() -> None:
    ticker = yf.Ticker("ISA.CL")
    results = {}
    for adjusted in (False, True):
        daily = ticker.history(
            period="max", interval="1d", auto_adjust=adjusted,
            actions=False, repair=False, timeout=30,
        )
        weekly = ticker.history(
            period="max", interval="1wk", auto_adjust=adjusted,
            actions=False, repair=False, timeout=30,
        )
        if daily.empty or weekly.empty:
            raise RuntimeError(
                f"Yahoo data missing: daily={len(daily)}, weekly={len(weekly)}, "
                f"auto_adjust={adjusted}"
            )
        suffix = "adjusted" if adjusted else "unadjusted"
        results[f"daily_resampled_{suffix}"] = summarize(
            f"daily_resampled_{suffix}", daily_to_weekly(daily)
        )
        results[f"native_weekly_{suffix}"] = summarize(
            f"native_weekly_{suffix}", native_weekly(weekly)
        )

    # Compare OHLCV for the last 8 overlapping weeks between aggregation methods.
    comparison = {}
    for adjusted in ("unadjusted", "adjusted"):
        d = results[f"daily_resampled_{adjusted}"]["last_8_bars"]
        w = results[f"native_weekly_{adjusted}"]["last_8_bars"]
        dm = {x["date"]: x for x in d}
        wm = {x["date"]: x for x in w}
        common = sorted(set(dm) & set(wm))
        comparison[adjusted] = [
            {
                "date": date,
                "daily_resampled": {k: dm[date][k] for k in ["open", "high", "low", "close", "volume"]},
                "native_weekly": {k: wm[date][k] for k in ["open", "high", "low", "close", "volume"]},
            }
            for date in common
        ]
    results["aggregation_comparison"] = comparison
    Path("isa_vfi_diagnostic.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
