import numpy as np
import pandas as pd


class RegimeAnalyzer:
    """Classify event dates using only information available at each date."""

    def __init__(self, lookback_weeks=26, bull_threshold=0.10, bear_threshold=-0.10):
        if lookback_weeks <= 0:
            raise ValueError("lookback_weeks must be positive.")
        if bear_threshold >= bull_threshold:
            raise ValueError("bear_threshold must be below bull_threshold.")
        self.lookback_weeks = int(lookback_weeks)
        self.bull_threshold = float(bull_threshold)
        self.bear_threshold = float(bear_threshold)

    def classify(self, benchmark):
        if "Close" not in benchmark.columns:
            raise ValueError("Column 'Close' is required.")
        df = benchmark.sort_index().copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Benchmark must use a DatetimeIndex.")

        prior = df["Close"].shift(self.lookback_weeks)
        df["Regime_Return"] = df["Close"] / prior - 1.0
        df["Market_Regime"] = np.select(
            [
                df["Regime_Return"] >= self.bull_threshold,
                df["Regime_Return"] <= self.bear_threshold,
            ],
            ["BULL", "BEAR"],
            default="SIDEWAYS",
        )
        return df

    def label_events(self, events, benchmark):
        if not isinstance(events.index, pd.DatetimeIndex):
            raise ValueError("Events must use a DatetimeIndex.")
        regime = self.classify(benchmark)
        labels = regime[["Regime_Return", "Market_Regime"]].reindex(events.index, method="ffill")
        result = events.copy()
        result["Market_Regime"] = labels["Market_Regime"]
        result["Regime_Return"] = labels["Regime_Return"]
        return result

    def summarize(self, events, return_column):
        if return_column not in events.columns:
            raise ValueError(f"Column '{return_column}' is required.")
        valid = events[events["Market_Regime"].notna() & events[return_column].notna()]
        if valid.empty:
            return pd.DataFrame()

        rows = []
        for regime, group in valid.groupby("Market_Regime"):
            returns = group[return_column]
            rows.append({
                "Market_Regime": regime,
                "Events": len(group),
                "Assets": group["Symbol"].nunique() if "Symbol" in group.columns else np.nan,
                "Average_Return": returns.mean(),
                "Median_Return": returns.median(),
                "Positive_Rate": returns.gt(0).mean() * 100.0,
                "Worst_Return": returns.min(),
                "Best_Return": returns.max(),
            })
        return pd.DataFrame(rows).sort_values("Average_Return", ascending=False).reset_index(drop=True)
