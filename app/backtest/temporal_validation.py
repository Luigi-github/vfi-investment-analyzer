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

    def summarize_walk_forward(self, windows):
        """Summarize out-of-sample consistency across walk-forward windows."""
        if not windows:
            return {
                "Windows": 0,
                "Valid_Windows": 0,
                "Average_Test_Return": float("nan"),
                "Median_Test_Return": float("nan"),
                "Positive_Window_Rate": float("nan"),
                "Worst_Test_Return": float("nan"),
                "Best_Test_Return": float("nan"),
            }

        returns = pd.Series(
            [
                window["test_summary"]["Average_Return"]
                for window in windows
            ],
            dtype="float64",
        ).dropna()

        if returns.empty:
            return {
                "Windows": len(windows),
                "Valid_Windows": 0,
                "Average_Test_Return": float("nan"),
                "Median_Test_Return": float("nan"),
                "Positive_Window_Rate": float("nan"),
                "Worst_Test_Return": float("nan"),
                "Best_Test_Return": float("nan"),
            }

        return {
            "Windows": len(windows),
            "Valid_Windows": len(returns),
            "Average_Test_Return": float(returns.mean()),
            "Median_Test_Return": float(returns.median()),
            "Positive_Window_Rate": float((returns > 0).mean() * 100.0),
            "Worst_Test_Return": float(returns.min()),
            "Best_Test_Return": float(returns.max()),
        }

    def summarize_benchmark_by_window(self, windows, benchmark_returns):
        """Compare each test window with a matching benchmark return."""
        if not windows:
            return pd.DataFrame()

        rows = []
        for number, window in enumerate(windows, start=1):
            test = window["test"]
            test_dates = test.index
            benchmark = benchmark_returns.reindex(test_dates).dropna()

            test_return = window["test_summary"]["Average_Return"]

            rows.append({
                "Window": number,
                "Cutoff": window["cutoff"],
                "Test_End": window["test_end"],
                "Test_Events": window["test_summary"]["Events"],
                "Test_Average_Return": test_return,
                "Benchmark_Average_Return": (
                    float(benchmark.mean())
                    if not benchmark.empty
                    else float("nan")
                ),
                "Excess_Average_Return": (
                    test_return - float(benchmark.mean())
                    if not benchmark.empty
                    else float("nan")
                ),
            })

        return pd.DataFrame(rows)

    def analyze(self, events, cutoff):
        train, test = self.split(events, cutoff)

        return {
            "cutoff": pd.Timestamp(cutoff),
            "train": train,
            "test": test,
            "train_summary": self.summarize(train),
            "test_summary": self.summarize(test),
        }
