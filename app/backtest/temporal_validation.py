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

        label_end = events.index + pd.to_timedelta(
            self.primary_horizon,
            unit="W"
        )

        train = events[label_end < cutoff].copy()
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


    def analyze_assets(self, asset_events, cutoff):
        rows = []

        for symbol, events in asset_events.items():
            result = self.analyze(events, cutoff)
            row = {"Symbol": symbol, "Cutoff": result["cutoff"]}

            for period in ("train", "test"):
                summary = result[f"{period}_summary"]
                prefix = period.title()
                row[f"{prefix}_Events"] = summary["Events"]
                row[f"{prefix}_Average_Return"] = summary["Average_Return"]
                row[f"{prefix}_Median_Return"] = summary["Median_Return"]
                row[f"{prefix}_Positive_Rate"] = summary["Positive_Rate"]

            rows.append(row)

        return pd.DataFrame(rows)

    def walk_forward(self, events, initial_train_weeks, test_weeks, step_weeks):
        if not isinstance(events.index, pd.DatetimeIndex):
            raise ValueError("Events must use a DatetimeIndex.")
        if initial_train_weeks <= 0 or test_weeks <= 0 or step_weeks <= 0:
            raise ValueError("Window sizes must be positive.")

        events = events.sort_index().copy()
        if events.empty:
            return []

        first_test = events.index.min() + pd.Timedelta(weeks=initial_train_weeks)
        windows = []
        cutoff = first_test

        while True:
            test_end = cutoff + pd.Timedelta(weeks=test_weeks)
            train, test = self.split(events, cutoff)
            test = test[test.index < test_end].copy()

            if test.empty:
                break

            windows.append({
                "cutoff": cutoff,
                "test_end": test_end,
                "train": train,
                "test": test,
                "train_summary": self.summarize(train),
                "test_summary": self.summarize(test),
            })

            cutoff = cutoff + pd.Timedelta(weeks=step_weeks)

        return windows

    def analyze(self, events, cutoff):
        train, test = self.split(events, cutoff)

        return {
            "cutoff": pd.Timestamp(cutoff),
            "train": train,
            "test": test,
            "train_summary": self.summarize(train),
            "test_summary": self.summarize(test),
        }
