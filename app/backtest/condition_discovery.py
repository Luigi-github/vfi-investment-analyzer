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

    def evaluate_selected(self, test_events, discovered):
        """Evaluate already-selected train hypotheses on untouched test data."""
        if discovered.empty:
            return discovered.copy()

        names = discovered["Condition"].tolist()
        missing = [name for name in names if name not in discovered["Condition"].values]
        if missing:
            raise ValueError(f"Unknown discovered conditions: {missing}")

        return self.evaluate_conditions(
            test_events,
            {
                name: discovered.loc[
                    discovered["Condition"] == name, "Condition"
                ].iloc[0]
                == name
                for name in names
            },
        )
