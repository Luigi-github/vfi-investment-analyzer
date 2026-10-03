import pandas as pd


class TemporalHoldoutAnalyzer:
    """Evaluate event outcomes before and after a fixed chronological cutoff."""

    def __init__(self, primary_horizon=26):
        self.primary_horizon = primary_horizon

    def split(self, events, cutoff):
        if events.empty:
            return events.copy(), events.copy()

        if not isinstance(events.index, pd.DatetimeIndex):
            raise ValueError("Events must use a DatetimeIndex.")

        cutoff = pd.Timestamp(cutoff)

        train = events[events.index < cutoff].copy()
        test = events[events.index >= cutoff].copy()

        return train, test

    def summarize(self, events):
        primary_column = f"Return_{self.primary_horizon}W"

        if primary_column not in events.columns:
            raise ValueError(
                f"Column '{primary_column}' is required."
            )

        valid = events[primary_column].dropna()

        if valid.empty:
            return {
                "Events": 0,
                "Average_Return": float("nan"),
                "Median_Return": float("nan"),
                "Positive_Rate": float("nan"),
            }

        return {
            "Events": int(len(valid)),
            "Average_Return": float(valid.mean()),
            "Median_Return": float(valid.median()),
            "Positive_Rate": float((valid > 0).mean() * 100.0),
        }

    def analyze(self, events, cutoff):
        train, test = self.split(events, cutoff)

        return {
            "cutoff": pd.Timestamp(cutoff),
            "train": train,
            "test": test,
            "train_summary": self.summarize(train),
            "test_summary": self.summarize(test),
        }
