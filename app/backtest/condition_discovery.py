import numpy as np
import pandas as pd


class ConditionDiscoveryAnalyzer:
    """Select rule hypotheses on train data and evaluate them on held-out data."""

    def __init__(
        self,
        primary_horizon=26,
        minimum_events=10,
        ranking_metric="Average_Return",
    ):
        if primary_horizon <= 0:
            raise ValueError("primary_horizon must be positive.")
        if minimum_events <= 0:
            raise ValueError("minimum_events must be positive.")
        if ranking_metric not in {
            "Average_Return",
            "Median_Return",
            "Positive_Rate",
        }:
            raise ValueError("Unsupported ranking_metric.")

        self.primary_horizon = primary_horizon
        self.minimum_events = minimum_events
        self.ranking_metric = ranking_metric

    def _primary_returns(self, events):
        column = f"Return_{self.primary_horizon}W"
        if column not in events.columns:
            raise ValueError(f"Column '{column}' is required.")
        return events[column]

    def evaluate_conditions(self, events, conditions):
        """Evaluate predefined conditions without selecting among them."""
        if events.empty:
            return pd.DataFrame()

        returns = self._primary_returns(events)
        valid = returns.notna()
        rows = []

        for name, mask in conditions.items():
            mask = pd.Series(mask, index=events.index).fillna(False).astype(bool)
            selected = events.loc[mask & valid]
            selected_returns = returns.loc[selected.index].dropna()

            count = len(selected_returns)
            rows.append({
                "Condition": name,
                "Count": count,
                "Positive_Rate": (
                    float((selected_returns > 0).mean() * 100.0)
                    if count else np.nan
                ),
                "Average_Return": (
                    float(selected_returns.mean())
                    if count else np.nan
                ),
                "Median_Return": (
                    float(selected_returns.median())
                    if count else np.nan
                ),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def build_standard_conditions(events):
        """Build the project's predefined VFI hypotheses from event data."""
        required = [
            "VFI",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Distance_Zero",
            "VFI_Range_Position",
        ]
        missing = [column for column in required if column not in events.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return {
            "VFI < 0": events["VFI"] < 0,
            "VFI < -5": events["VFI"] < -5,
            "VFI near zero": events["VFI_Distance_Zero"] <= 1,
            "Slope > 0": events["VFI_Slope"] > 0,
            "Slope 4W > 0": events["VFI_Slope_4W"] > 0,
            "Slope 8W > 0": events["VFI_Slope_8W"] > 0,
            "Range position <= 20%": events["VFI_Range_Position"] <= 0.20,
            "Slope > 0 AND Slope 8W > 0": (
                (events["VFI_Slope"] > 0)
                & (events["VFI_Slope_8W"] > 0)
            ),
            "Slope 4W > 0 AND Slope 8W > 0": (
                (events["VFI_Slope_4W"] > 0)
                & (events["VFI_Slope_8W"] > 0)
            ),
            "VFI < 0 AND Slope > 0": (
                (events["VFI"] < 0)
                & (events["VFI_Slope"] > 0)
            ),
            "VFI < -5 AND Slope > 0": (
                (events["VFI"] < -5)
                & (events["VFI_Slope"] > 0)
            ),
            "VFI < 0 AND Slope > 0 AND Slope 8W > 0": (
                (events["VFI"] < 0)
                & (events["VFI_Slope"] > 0)
                & (events["VFI_Slope_8W"] > 0)
            ),
        }

    def discover(self, train_events, conditions):
        """Rank hypotheses using train data only."""
        result = self.evaluate_conditions(train_events, conditions)

        if result.empty:
            return result

        eligible = result[result["Count"] >= self.minimum_events].copy()

        if eligible.empty:
            return eligible

        eligible = eligible.sort_values(
            [self.ranking_metric, "Count"],
            ascending=[False, False],
        ).reset_index(drop=True)

        eligible["Train_Selected"] = True
        return eligible

    def walk_forward_discovery(
        self,
        events,
        initial_train_weeks,
        test_weeks,
        step_weeks,
        condition_builder=None,
        top_n=1,
    ):
        """Discover conditions in each train window and evaluate them in test."""
        if not isinstance(events.index, pd.DatetimeIndex):
            raise ValueError("Events must use a DatetimeIndex.")
        if initial_train_weeks <= 0 or test_weeks <= 0 or step_weeks <= 0:
            raise ValueError("Window sizes must be positive.")
        if top_n <= 0:
            raise ValueError("top_n must be positive.")

        events = events.sort_index().copy()
        if events.empty:
            return []

        if condition_builder is None:
            condition_builder = self.build_standard_conditions

        first_test = (
            events.index.min()
            + pd.Timedelta(weeks=initial_train_weeks)
        )
        cutoff = first_test
        windows = []

        while True:
            test_end = cutoff + pd.Timedelta(weeks=test_weeks)
            label_end = (
                events.index
                + pd.Timedelta(weeks=self.primary_horizon)
            )
            train = events[label_end < cutoff].copy()
            test = events[
                (events.index >= cutoff)
                & (events.index < test_end)
            ].copy()

            if test.empty:
                break

            train_conditions = condition_builder(train)
            discovered = self.discover(train, train_conditions)
            selected = discovered.head(top_n).copy()

            test_conditions = condition_builder(test)
            test_results = self.evaluate_selected(
                test,
                selected,
                test_conditions,
            )

            windows.append({
                "cutoff": cutoff,
                "test_end": test_end,
                "train": train,
                "test": test,
                "discovered": discovered,
                "selected": selected,
                "test_results": test_results,
            })

            cutoff = cutoff + pd.Timedelta(weeks=step_weeks)

        return windows

    def summarize_walk_forward(self, windows):
        """Summarize out-of-sample results from selected hypotheses."""
        rows = []

        for number, window in enumerate(windows, start=1):
            selected = window["selected"]
            test_results = window["test_results"]

            if selected.empty or test_results.empty:
                rows.append({
                    "Window": number,
                    "Cutoff": window["cutoff"],
                    "Test_End": window["test_end"],
                    "Selected_Condition": None,
                    "Train_Count": 0,
                    "Train_Average_Return": np.nan,
                    "Test_Count": 0,
                    "Test_Average_Return": np.nan,
                    "Test_Median_Return": np.nan,
                    "Test_Positive_Rate": np.nan,
                })
                continue

            best = selected.iloc[0]
            evaluated = test_results.iloc[0]

            rows.append({
                "Window": number,
                "Cutoff": window["cutoff"],
                "Test_End": window["test_end"],
                "Selected_Condition": best["Condition"],
                "Train_Count": int(best["Count"]),
                "Train_Average_Return": float(best["Average_Return"]),
                "Test_Count": int(evaluated["Count"]),
                "Test_Average_Return": float(evaluated["Average_Return"]),
                "Test_Median_Return": float(evaluated["Median_Return"]),
                "Test_Positive_Rate": float(evaluated["Positive_Rate"]),
            })

        return pd.DataFrame(rows)

    def evaluate_selected(self, test_events, discovered, conditions):
        """Evaluate train-selected hypotheses on untouched test data."""
        if discovered.empty:
            return discovered.copy()

        names = discovered["Condition"].tolist()
        missing = [name for name in names if name not in conditions]
        if missing:
            raise ValueError(f"Missing condition definitions: {missing}")

        selected_conditions = {
            name: conditions[name]
            for name in names
        }
        return self.evaluate_conditions(test_events, selected_conditions)
