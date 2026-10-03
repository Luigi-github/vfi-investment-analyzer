import pandas as pd

from app.backtest.execution_costs import LongOnlyExecutionCostModel


class BenchmarkAnalyzer:
    """Compare event outcomes with an unconditional forward-return baseline."""

    def __init__(
        self,
        horizons=(4, 8, 12, 26, 52),
        execution_cost_model=None,
    ):
        self.horizons = tuple(horizons)

        if execution_cost_model is not None and not isinstance(
            execution_cost_model, LongOnlyExecutionCostModel
        ):
            raise TypeError(
                "execution_cost_model must be a LongOnlyExecutionCostModel or None."
            )

        self.execution_cost_model = execution_cost_model

    def calculate_forward_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        if "Close" not in data.columns:
            raise ValueError("Column 'Close' is required.")

        df = data.copy()

        for horizon in self.horizons:
            gross = (
                (df["Close"].shift(-horizon) / df["Close"]) - 1.0
            ) * 100.0

            df[f"Return_{horizon}W"] = gross

            if self.execution_cost_model is not None:
                df[f"Net_Return_{horizon}W"] = (
                    self.execution_cost_model.net_return(gross)
                )

        return df

    def compare_event_dates(
        self,
        data: pd.DataFrame,
        event_dates,
    ) -> pd.DataFrame:
        df = self.calculate_forward_returns(data)
        event_index = pd.DatetimeIndex(event_dates)
        event_index = event_index.intersection(df.index)

        rows = []

        for horizon in self.horizons:
            return_column = f"Return_{horizon}W"

            benchmark_returns = df[return_column].dropna()

            event_returns = (
                df.loc[event_index, return_column]
                .dropna()
            )

            if self.execution_cost_model is not None:
                net_column = f"Net_Return_{horizon}W"
                benchmark_returns = df[net_column].dropna()
                event_returns = (
                    df.loc[event_index, net_column]
                    .dropna()
                )

            if benchmark_returns.empty:
                continue

            rows.append(
                {
                    "Horizon": f"{horizon}W",
                    "Event_Count": len(event_returns),
                    "Benchmark_Count": len(benchmark_returns),
                    "Event_Average_Return": event_returns.mean(),
                    "Benchmark_Average_Return": benchmark_returns.mean(),
                    "Excess_Average_Return": (
                        event_returns.mean() - benchmark_returns.mean()
                        if not event_returns.empty
                        else float("nan")
                    ),
                    "Event_Positive_Rate": (
                        event_returns.gt(0).mean() * 100
                        if not event_returns.empty
                        else float("nan")
                    ),
                    "Benchmark_Positive_Rate": (
                        benchmark_returns.gt(0).mean() * 100
                    ),
                }
            )

        return pd.DataFrame(rows)
