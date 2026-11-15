from __future__ import annotations

"""Compare ISA weekly LazyBear VFI using Yahoo adjusted vs unadjusted OHLCV."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from app.indicators.vfi import VFIIndicator


def weekly_bars(data: pd.DataFrame) -> pd.DataFrame:
    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.to_datetime(data.index, errors="coerce")
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    data.index = idx.normalize()
    data = data.loc[~data.index.isna()].sort_index()
    bars = data.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open", "High", "Low", "Close"])
    # Reference screenshot is dated 2026-11-14; use last completed Friday.
    bars = bars.loc[bars.index <= pd.Timestamp("2026-11-13")]
    return bars


def main() -> None:
    ticker = yf.Ticker("ISA.CL")
    results = {}
    for adjusted in (False, True):
        daily = ticker.history(
            period="max", interval="1d", auto_adjust=adjusted,
            actions=False, repair=False, timeout=30,
        )
        if daily.empty:
            raise RuntimeError(f"Yahoo returned no data for ISA.CL (auto_adjust={adjusted})")
        weekly = weekly_bars(daily)
        indicator = VFIIndicator(length=130, coef=0.2, vcoef=2.5, signal_length=5)
        calculated = indicator.calculate(weekly)
        latest = calculated.dropna(subset=["VFI"]).iloc[-1]
        results["adjusted" if adjusted else "unadjusted"] = {
            "rows_daily": int(len(daily)),
            "rows_weekly": int(len(weekly)),
            "first_week": weekly.index.min().date().isoformat(),
            "last_week": weekly.index.max().date().isoformat(),
            "weekly_close": float(latest["Close"]),
            "weekly_volume": float(latest["Volume"]),
            "typical_price": float(latest["TypicalPrice"]),
            "volatility": float(latest["Volatility"]),
            "cutoff": float(latest["Cutoff"]),
            "volume_average": float(latest["VolumeAverage"]),
            "volume_max": float(latest["VolumeMax"]),
            "capped_volume": float(latest["CappedVolume"]),
            "money_flow": float(latest["MoneyFlow"]),
            "price_movement": int(latest["PriceMovement"]),
            "volume_flow": float(latest["VolumeFlow"]),
            "rolling_volume_flow": float(latest["RollingVolumeFlow"]),
            "vfi": float(latest["VFI"]),
            "signal_5": float(latest["VFI_Signal_5"]),
            "bars": [
                {
                    "date": index.date().isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                    "vfi": None if pd.isna(row["VFI"]) else float(row["VFI"]),
                    "signal_5": None if pd.isna(row["VFI_Signal_5"]) else float(row["VFI_Signal_5"]),
                }
                for index, row in calculated.tail(8).iterrows()
            ],
        }
    Path("isa_vfi_diagnostic.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
