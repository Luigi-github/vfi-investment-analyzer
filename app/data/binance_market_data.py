"""Binance public market-data services for Spot and USDⓈ-M perpetual futures."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Iterable

import pandas as pd
import requests


class BinanceMarketDataService:
    SPOT_BASE = "https://data-api.binance.vision"
    FUTURES_BASE = "https://fapi.binance.com"
    PROVIDER = "Binance Spot + USDⓈ-M Futures"
    KLINE_COLUMNS = ["OpenTime", "Open", "High", "Low", "Close", "Volume",
                     "CloseTime", "QuoteVolume", "Trades", "TakerBuyBaseVolume",
                     "TakerBuyQuoteVolume", "Ignore"]

    def __init__(self, timeout: int = 20, sleep_seconds: float = 0.05):
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.last_metadata = {}

    def _get(self, base: str, path: str, params: dict) -> list | dict:
        response = self.session.get(base + path, params=params, timeout=self.timeout)
        if response.status_code == 429:
            time.sleep(1.0)
            response = self.session.get(base + path, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def exchange_info(self, market: str = "spot") -> dict:
        if market == "spot":
            return self._get(self.SPOT_BASE, "/api/v3/exchangeInfo", {})
        if market == "futures":
            return self._get(self.FUTURES_BASE, "/fapi/v1/exchangeInfo", {})
        raise ValueError("market must be 'spot' or 'futures'")

    def discover_usdt_perpetuals(
        self,
        min_quote_volume_usd: float = 5_000_000,
        limit: int | None = None,
    ) -> list[str]:
        info = self.exchange_info("futures")
        symbols = {
            item["symbol"]
            for item in info.get("symbols", [])
            if item.get("status") == "TRADING"
            and item.get("contractType") == "PERPETUAL"
            and item.get("quoteAsset") == "USDT"
        }
        tickers = self._get(self.FUTURES_BASE, "/fapi/v1/ticker/24hr", {})
        ranked = sorted(
            (
                (row["symbol"], float(row.get("quoteVolume", 0)))
                for row in tickers
                if row.get("symbol") in symbols
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        result = [symbol for symbol, volume in ranked if volume >= min_quote_volume_usd]
        return result[:limit] if limit else result

    def discover_spot_usdt(
        self,
        min_quote_volume_usd: float = 5_000_000,
        limit: int | None = None,
    ) -> list[str]:
        info = self.exchange_info("spot")
        symbols = {
            item["symbol"]
            for item in info.get("symbols", [])
            if item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed", True)
        }
        tickers = self._get(self.SPOT_BASE, "/api/v3/ticker/24hr", {})
        ranked = sorted(
            (
                (row["symbol"], float(row.get("quoteVolume", 0)))
                for row in tickers
                if row.get("symbol") in symbols
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        result = [symbol for symbol, volume in ranked if volume >= min_quote_volume_usd]
        return result[:limit] if limit else result

    @staticmethod
    def _frame(rows: list) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows, columns=BinanceMarketDataService.KLINE_COLUMNS)
        numeric = ["Open", "High", "Low", "Close", "Volume", "QuoteVolume",
                   "Trades", "TakerBuyBaseVolume", "TakerBuyQuoteVolume"]
        for col in numeric:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.index = pd.to_datetime(frame["OpenTime"], unit="ms", utc=True)
        frame.index.name = "Date"
        frame = frame.drop(columns=["OpenTime", "CloseTime", "Ignore"])
        return frame.sort_index()

    def get_klines(
        self,
        symbol: str,
        interval: str,
        market: str = "spot",
        start_time: str | None = None,
        end_time: str | None = None,
        max_bars: int | None = None,
    ) -> pd.DataFrame:
        base = self.SPOT_BASE if market == "spot" else self.FUTURES_BASE
        path = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
        start_ms = int(pd.Timestamp(start_time, tz="UTC").timestamp() * 1000) if start_time else None
        end_ms = int(pd.Timestamp(end_time, tz="UTC").timestamp() * 1000) if end_time else None
        rows: list = []
        while True:
            params = {"symbol": symbol, "interval": interval, "limit": 1000}
            if start_ms is not None:
                params["startTime"] = start_ms
            if end_ms is not None:
                params["endTime"] = end_ms
            batch = self._get(base, path, params)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1000 or (max_bars and len(rows) >= max_bars):
                break
            next_start = int(batch[-1][0]) + 1
            if start_ms is not None and next_start <= start_ms:
                break
            start_ms = next_start
            time.sleep(self.sleep_seconds)
        frame = self._frame(rows)
        if max_bars:
            frame = frame.iloc[-max_bars:]
        frame = frame[~frame.index.duplicated(keep="last")]
        self.last_metadata = {
            "Symbol": symbol,
            "Market": market,
            "Interval": interval,
            "Provider": self.PROVIDER,
            "Retrieved_At_UTC": pd.Timestamp.now(tz="UTC").isoformat(),
            "Rows": int(len(frame)),
            "First_Date": frame.index.min().isoformat() if not frame.empty else None,
            "Last_Date": frame.index.max().isoformat() if not frame.empty else None,
            "Data_SHA256": self._hash_frame(frame),
        }
        return frame

    def get_open_interest_history(
        self,
        symbol: str,
        period: str = "1h",
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> pd.DataFrame:
        params = {"symbol": symbol, "period": period, "limit": 500}
        if start_time:
            params["startTime"] = int(pd.Timestamp(start_time, tz="UTC").timestamp() * 1000)
        if end_time:
            params["endTime"] = int(pd.Timestamp(end_time, tz="UTC").timestamp() * 1000)
        rows = self._get(self.FUTURES_BASE, "/futures/data/openInterestHist", params)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        for col in ("sumOpenInterest", "sumOpenInterestValue"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame.index.name = "Date"
        return frame.drop(columns=["timestamp"], errors="ignore").sort_index()

    def get_funding_history(
        self,
        symbol: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> pd.DataFrame:
        params = {"symbol": symbol, "limit": 1000}
        if start_time:
            params["startTime"] = int(pd.Timestamp(start_time, tz="UTC").timestamp() * 1000)
        if end_time:
            params["endTime"] = int(pd.Timestamp(end_time, tz="UTC").timestamp() * 1000)
        rows = self._get(self.FUTURES_BASE, "/fapi/v1/fundingRate", params)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
        frame["markPrice"] = pd.to_numeric(frame["markPrice"], errors="coerce")
        frame.index = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
        frame.index.name = "Date"
        return frame.sort_index()

    @staticmethod
    def _hash_frame(data: pd.DataFrame) -> str:
        raw = pd.util.hash_pandas_object(data.sort_index(), index=True).values.tobytes()
        return hashlib.sha256(raw).hexdigest()

    def get_last_metadata(self) -> dict:
        return dict(self.last_metadata)

    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, rule: str) -> pd.DataFrame:
        aggregations = {
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
            "QuoteVolume": "sum", "Trades": "sum",
            "TakerBuyBaseVolume": "sum", "TakerBuyQuoteVolume": "sum",
        }
        available = {k: v for k, v in aggregations.items() if k in data.columns}
        return data.resample(rule).agg(available).dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    def get_multi_timeframe(
        self,
        symbol: str,
        market: str = "spot",
        limits: dict[str, int] | None = None,
    ) -> dict[str, pd.DataFrame]:
        limits = limits or {"1w": 400, "1d": 900, "4h": 1400, "1h": 3000}
        return {
            "1W": self.get_klines(symbol, "1w", market, max_bars=limits["1w"]),
            "1D": self.get_klines(symbol, "1d", market, max_bars=limits["1d"]),
            "4H": self.get_klines(symbol, "4h", market, max_bars=limits["4h"]),
            "1H": self.get_klines(symbol, "1h", market, max_bars=limits["1h"]),
        }
