"""Free Yahoo Finance collector with a persistent local cache and quality audit.

Daily candles are never fabricated. Missing daily records remain missing; weekly
imputation is a separate research transformation. Cached prices are never
considered current unless their source date passes the freshness check.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
OPTIONAL_COLUMNS = ["Adj Close", "Dividends", "Stock Splits"]
STORE_COLUMNS = PRICE_COLUMNS + OPTIONAL_COLUMNS
PROVIDER = "Yahoo Finance via yfinance (free; unofficial feed)"


def safe_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", symbol.strip())


class ColombiaMarketDataStore:
    """Incremental CSV cache for Colombia tickers with per-refresh audit files."""

    def __init__(self, root: str | Path = "data/market_cache/colombia"):
        self.root = Path(root)
        self.daily_dir = self.root / "daily"
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.root / "source_audit.csv"
        self.quotes_path = self.root / "latest_quotes.csv"

    def cache_path(self, symbol: str) -> Path:
        return self.daily_dir / f"{safe_symbol(symbol)}.csv"

    def load_cached(self, symbol: str) -> pd.DataFrame:
        path = self.cache_path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=STORE_COLUMNS)
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        for col in STORE_COLUMNS:
            if col not in frame.columns:
                frame[col] = 0.0 if col in ("Dividends", "Stock Splits") else float("nan")
        return frame[STORE_COLUMNS]

    @staticmethod
    def _normalize_download(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            raise ValueError(f"{symbol}: Yahoo Finance returned no rows")
        data = frame.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(col[0]) for col in data.columns]
        missing = [col for col in PRICE_COLUMNS if col not in data.columns]
        if missing:
            raise ValueError(f"{symbol}: missing required Yahoo columns: {missing}")
        for col in OPTIONAL_COLUMNS:
            if col not in data.columns:
                data[col] = 0.0 if col in ("Dividends", "Stock Splits") else float("nan")
        data = data[STORE_COLUMNS].copy()
        data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data.index = pd.DatetimeIndex(data.index).normalize()
        data.index.name = "Date"
        data = data[~data.index.duplicated(keep="last")].sort_index()
        for col in STORE_COLUMNS:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        return data

    @staticmethod
    def validate(
        data: pd.DataFrame,
        *,
        as_of: date | pd.Timestamp | None = None,
        max_stale_calendar_days: int = 7,
    ) -> dict:
        today = pd.Timestamp(as_of or pd.Timestamp.now().normalize()).normalize()
        if data.empty:
            return {
                "Valid": False, "Fresh": False, "Status": "NO_DATA",
                "Rows": 0, "Last_Date": None, "Age_Calendar_Days": None,
                "Issues": "NO_DATA",
            }
        frame = data.copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        issues: list[str] = []
        missing = [col for col in PRICE_COLUMNS if col not in frame.columns]
        if missing:
            issues.append("MISSING_COLUMNS:" + ",".join(missing))
        else:
            numeric = frame[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
            if numeric.isna().any().any():
                issues.append("NULL_OR_NON_NUMERIC_OHLCV")
            if (numeric[["Open", "High", "Low", "Close"]] <= 0).any().any():
                issues.append("NON_POSITIVE_PRICE")
            if (numeric["Volume"] < 0).any():
                issues.append("NEGATIVE_VOLUME")
            if (numeric["High"] < numeric[["Open", "Low", "Close"]].max(axis=1)).any():
                issues.append("HIGH_BELOW_OHLC")
            if (numeric["Low"] > numeric[["Open", "High", "Close"]].min(axis=1)).any():
                issues.append("LOW_ABOVE_OHLC")
        if frame.index.has_duplicates:
            issues.append("DUPLICATE_DATES")
        if not frame.index.is_monotonic_increasing:
            issues.append("UNSORTED_DATES")
        last_date = pd.Timestamp(frame.index.max()).normalize()
        age = max(0, int((today - last_date).days))
        fresh = age <= max_stale_calendar_days
        if not fresh:
            issues.append("STALE_LAST_OBSERVATION")
        invalid_prefixes = (
            "MISSING_COLUMNS", "NULL_OR_NON_NUMERIC", "NON_POSITIVE",
            "NEGATIVE", "HIGH_BELOW", "LOW_ABOVE", "DUPLICATE", "UNSORTED",
        )
        valid = not any(item.startswith(invalid_prefixes) for item in issues)
        status = "OK" if valid and fresh else ("STALE" if valid else "INVALID")
        return {
            "Valid": valid, "Fresh": fresh, "Status": status,
            "Rows": int(len(frame)), "Last_Date": last_date.date().isoformat(),
            "Age_Calendar_Days": age, "Issues": "|".join(issues),
        }

    @staticmethod
    def _fetch(symbol: str, start: pd.Timestamp | None = None) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        if start is None:
            downloaded = ticker.history(
                period="max", interval="1d", auto_adjust=False, actions=True
            )
        else:
            downloaded = ticker.history(
                start=start.date().isoformat(), end=end.date().isoformat(),
                interval="1d", auto_adjust=False, actions=True,
            )
        return ColombiaMarketDataStore._normalize_download(downloaded, symbol)

    def refresh_symbol(
        self, symbol: str, *, max_stale_calendar_days: int = 7,
        sleep_seconds: float = 0.25,
    ) -> dict:
        retrieved_at = pd.Timestamp.now(tz="UTC").isoformat()
        cached = self.load_cached(symbol)
        try:
            start = cached.index.max() - pd.Timedelta(days=10) if not cached.empty else None
            fresh_rows = self._fetch(symbol, start=start)
            combined = pd.concat([cached, fresh_rows])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined = combined[STORE_COLUMNS]
            quality = self.validate(combined, max_stale_calendar_days=max_stale_calendar_days)
            if not quality["Valid"]:
                raise ValueError(
                    f"{symbol}: refusing to overwrite cache with invalid OHLCV: "
                    f"{quality['Issues']}"
                )
            # Only replace the persisted cache after the complete merged series
            # passes OHLCV validation; malformed provider responses must not
            # destroy a previously usable history.
            combined.to_csv(self.cache_path(symbol), index_label="Date", date_format="%Y-%m-%d")
            result = {
                "Symbol": symbol, "Provider": PROVIDER, "Retrieved_At_UTC": retrieved_at,
                **quality, "Cache_File": str(self.cache_path(symbol)), "Error": "",
            }
            if sleep_seconds:
                time.sleep(sleep_seconds)
            return result
        except Exception as exc:
            quality = self.validate(cached, max_stale_calendar_days=max_stale_calendar_days)
            return {
                "Symbol": symbol, "Provider": PROVIDER, "Retrieved_At_UTC": retrieved_at,
                **quality, "Cache_File": str(self.cache_path(symbol)),
                "Error": f"{type(exc).__name__}: {exc}",
            }

    def refresh_many(
        self, symbols: Iterable[str], *, max_stale_calendar_days: int = 7,
        sleep_seconds: float = 0.25,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        audits, quotes = [], []
        for symbol in dict.fromkeys(symbols):
            audit = self.refresh_symbol(
                symbol, max_stale_calendar_days=max_stale_calendar_days,
                sleep_seconds=sleep_seconds,
            )
            audits.append(audit)
            data = self.load_cached(symbol)
            if not data.empty:
                latest = data.iloc[-1]
                quotes.append({
                    "Symbol": symbol, "Date": data.index[-1].date().isoformat(),
                    "Close": latest.get("Close"), "Volume": latest.get("Volume"),
                    "Provider": PROVIDER, "Status": audit["Status"],
                    "Fresh": audit["Fresh"], "Age_Calendar_Days": audit["Age_Calendar_Days"],
                    "Error": audit["Error"],
                })
            else:
                quotes.append({
                    "Symbol": symbol, "Date": None, "Close": None, "Volume": None,
                    "Provider": PROVIDER, "Status": audit["Status"], "Fresh": audit["Fresh"],
                    "Age_Calendar_Days": audit["Age_Calendar_Days"], "Error": audit["Error"],
                })
        audit_frame, quote_frame = pd.DataFrame(audits), pd.DataFrame(quotes)
        self.root.mkdir(parents=True, exist_ok=True)
        audit_frame.to_csv(self.audit_path, index=False)
        quote_frame.to_csv(self.quotes_path, index=False)
        valid_fresh = int(((audit_frame["Valid"] == True) & (audit_frame["Fresh"] == True)).sum()) if not audit_frame.empty else 0
        (self.root / "metadata.json").write_text(json.dumps({
            "Provider": PROVIDER,
            "Updated_At_UTC": pd.Timestamp.now(tz="UTC").isoformat(),
            "Symbols_Requested": len(audit_frame),
            "Symbols_Valid_And_Fresh": valid_fresh,
            "Note": "Free/unofficial Yahoo feed. Validate against BVC/SFC before trading; no daily bars are imputed.",
        }, indent=2), encoding="utf-8")
        return audit_frame, quote_frame
