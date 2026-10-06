import numpy as np
import pandas as pd


REQUIRED_SIGNAL_COLUMNS = {"Symbol"}


class SignalOutcomeBuilder:
    """Canonical signal -> executable trade -> outcome representation."""

    def __init__(self, horizon_weeks=26):
        if horizon_weeks <= 0:
            raise ValueError("horizon_weeks must be positive.")
        self.horizon_weeks = int(horizon_weeks)

    def build(self, prices, signals):
        if not isinstance(signals.index, pd.DatetimeIndex):
            raise ValueError("signals must use a DatetimeIndex.")
        if not REQUIRED_SIGNAL_COLUMNS.issubset(signals.columns):
            raise ValueError("signals must contain Symbol.")

        rows = []
        for signal_date, signal in signals.sort_index().iterrows():
            symbol = signal["Symbol"]
            if symbol not in prices:
                continue
            frame = prices[symbol].sort_index()
            dates = pd.DatetimeIndex(frame.index)
            entry_pos = dates.searchsorted(signal_date, side="right")
            if entry_pos >= len(dates):
                continue
            exit_pos = entry_pos + self.horizon_weeks
            if exit_pos >= len(dates):
                continue

            entry_date, exit_date = dates[entry_pos], dates[exit_pos]
            entry_price = float(frame.loc[entry_date, "Close"])
            exit_price = float(frame.loc[exit_date, "Close"])
            row = {
                "Signal_ID": f"{symbol}_{signal_date.date()}_{signal.name}",
                "Symbol": symbol,
                "Signal_Date": signal_date,
                "Entry_Date": entry_date,
                "Exit_Date": exit_date,
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "Gross_Price_Return_Pct": (exit_price / entry_price - 1.0) * 100.0,
            }
            if "Adj Close" in frame.columns:
                entry_adj = float(frame.loc[entry_date, "Adj Close"])
                exit_adj = float(frame.loc[exit_date, "Adj Close"])
                row["Gross_Total_Return_Pct"] = (exit_adj / entry_adj - 1.0) * 100.0

            for column in ("VFI", "VFI_Slope", "VFI_Slope_4W", "VFI_Slope_8W", "VFI_Cycle_Stage", "Signal_Score"):
                if column in signal.index:
                    row[column] = signal[column]
            rows.append(row)

        return pd.DataFrame(rows)
