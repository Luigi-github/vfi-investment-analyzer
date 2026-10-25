import hashlib
import json

import yfinance as yf
import pandas as pd


class MarketDataService:
    """Download market data and retain reproducibility metadata."""

    PROVIDER = "Yahoo Finance / yfinance"

    def __init__(self):
        self.last_metadata = {}

    @staticmethod
    def _hash_frame(data: pd.DataFrame) -> str:
        payload = data.copy()
        payload.index = pd.DatetimeIndex(payload.index)
        payload = payload.sort_index()
        raw = pd.util.hash_pandas_object(payload, index=True).values.tobytes()
        return hashlib.sha256(raw).hexdigest()

    def get_daily_data(
        self,
        symbol: str,
        period: str = "5y"
    ) -> pd.DataFrame:
        data = yf.Ticker(symbol).history(
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=True,
        )

        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

        columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Dividends",
            "Stock Splits",
        ]
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise ValueError(f"Missing Yahoo Finance columns for {symbol}: {missing}")

        data = data[columns].copy()
        data.index = pd.to_datetime(data.index)

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data = data[~data.index.duplicated(keep="last")].sort_index()
        today = pd.Timestamp.now().normalize()
        data = data[data.index.normalize() <= today]

        self.last_metadata = {
            "Symbol": symbol,
            "Provider": self.PROVIDER,
            "Retrieved_At_UTC": pd.Timestamp.now(tz="UTC").isoformat(),
            "Period": period,
            "Rows": int(len(data)),
            "First_Date": data.index.min().isoformat() if not data.empty else None,
            "Last_Date": data.index.max().isoformat() if not data.empty else None,
            "Currency": None,
            "Timezone": str(data.index.tz) if data.index.tz is not None else "naive",
            "Data_SHA256": self._hash_frame(data),
        }
        return data

    def get_last_metadata(self) -> dict:
        return dict(self.last_metadata)

    def convert_to_weekly(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily data into Friday-ended weeks and fill missing weeks.

        If a week has no market data, repeat the prior week's OHLCV values so
        indicators use a continuous weekly timeline. Corporate-action fields
        are set to zero on synthetic weeks to avoid duplicating dividends or
        stock splits.
        """
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        aggregations = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
        if "Adj Close" in data.columns:
            aggregations["Adj Close"] = "last"
        if "Dividends" in data.columns:
            aggregations["Dividends"] = "sum"
        if "Stock Splits" in data.columns:
            aggregations["Stock Splits"] = "sum"

        weekly = data.resample("W-FRI").agg(aggregations).dropna(
            subset=["Open", "High", "Low", "Close", "Volume"]
        )

        today = pd.Timestamp.now().normalize()
        current_friday = today + pd.Timedelta(days=(4 - today.weekday()) % 7)
        weekly = weekly[weekly.index < current_friday]

        if not weekly.empty:
            full_index = pd.date_range(
                start=weekly.index.min(),
                end=weekly.index.max(),
                freq="W-FRI",
                name=weekly.index.name,
            )
            missing_weeks = ~full_index.isin(weekly.index)
            weekly = weekly.reindex(full_index).ffill()

            # A repeated OHLCV row is synthetic; don't repeat one-off actions.
            for column in ("Dividends", "Stock Splits"):
                if column in weekly.columns:
                    weekly.loc[missing_weeks, column] = 0

        return weekly

    @staticmethod
    def metadata_json(metadata: dict) -> str:
        return json.dumps(metadata, sort_keys=True, indent=2, default=str)
