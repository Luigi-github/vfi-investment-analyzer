import numpy as np
import pandas as pd


class LiquidityAnalyzer:
    """Point-in-time liquidity and position-capacity checks."""

    def __init__(self, lookback_weeks=13):
        if lookback_weeks <= 1:
            raise ValueError("lookback_weeks must be greater than 1.")
        self.lookback_weeks = int(lookback_weeks)

    def metrics_at(self, data, date, capital, max_participation_pct):
        if "Close" not in data.columns or "Volume" not in data.columns:
            raise ValueError("Close and Volume columns are required.")

        date = pd.Timestamp(date)
        frame = data.loc[:date, ["Close", "Volume"]].copy()
        frame["DollarVolume"] = (
            pd.to_numeric(frame["Close"], errors="coerce")
            * pd.to_numeric(frame["Volume"], errors="coerce")
        )
        recent = frame["DollarVolume"].dropna().tail(self.lookback_weeks)
        if recent.empty:
            return {
                "Avg_Dollar_Volume": np.nan,
                "Max_Tradable_Notional": np.nan,
                "Max_Tradable_Weight": np.nan,
                "Liquidity_Status": "NO_DATA",
            }

        avg_dollar_volume = float(recent.mean())
        max_notional = avg_dollar_volume * float(max_participation_pct) / 100.0
        max_weight = max_notional / float(capital) if capital > 0 else np.nan

        return {
            "Avg_Dollar_Volume": avg_dollar_volume,
            "Max_Tradable_Notional": max_notional,
            "Max_Tradable_Weight": max_weight,
            "Liquidity_Status": "OK" if max_weight > 0 else "ILLIQUID",
        }

    def annotate_signals(
        self,
        signals,
        asset_data,
        capital,
        max_participation_pct=1.0,
        min_dollar_volume=0.0,
        target_position_weight=None,
    ):
        if capital <= 0:
            raise ValueError("capital must be positive.")
        if max_participation_pct <= 0:
            raise ValueError("max_participation_pct must be positive.")

        result = signals.copy()
        metrics = []

        for date, row in result.iterrows():
            symbol = row["Symbol"]
            if symbol not in asset_data:
                raise ValueError(f"Missing asset data for {symbol}.")
            metric = self.metrics_at(
                asset_data[symbol],
                date,
                capital,
                max_participation_pct,
            )
            metric["Date"] = date
            metric["Symbol"] = symbol
            metrics.append(metric)

        metric_frame = pd.DataFrame(metrics).set_index(["Date", "Symbol"])
        joined = result.reset_index(names="Date").set_index(["Date", "Symbol"])
        joined = joined.join(metric_frame, how="left").reset_index()

        if min_dollar_volume > 0:
            joined["Liquidity_Status"] = np.where(
                joined["Avg_Dollar_Volume"] >= min_dollar_volume,
                joined["Liquidity_Status"],
                "BELOW_MIN_DOLLAR_VOLUME",
            )

        if target_position_weight is not None:
            joined["Liquidity_Status"] = np.where(
                joined["Max_Tradable_Weight"] >= target_position_weight,
                joined["Liquidity_Status"],
                "INSUFFICIENT_CAPACITY",
            )

        joined["Liquidity_Eligible"] = joined["Liquidity_Status"].eq("OK")
        return joined

    def filter_signals(self, *args, **kwargs):
        annotated = self.annotate_signals(*args, **kwargs)
        return annotated[annotated["Liquidity_Eligible"]].copy()
