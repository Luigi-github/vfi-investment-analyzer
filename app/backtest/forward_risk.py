import numpy as np
import pandas as pd


class ForwardRiskAnalyzer:
    """Calculate forward outcome/risk metrics without using them as signals."""

    def __init__(self, horizons=(4, 8, 12, 26, 52)):
        self.horizons = tuple(horizons)

        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("All horizons must be positive integers.")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "Close" not in data.columns:
            raise ValueError("Column 'Close' is required.")

        df = data.copy()
        prices = pd.to_numeric(df["Close"], errors="coerce")

        for horizon in self.horizons:
            max_favorable = []
            max_adverse = []
            max_drawdown = []

            values = prices.to_numpy(dtype=float)

            for position in range(len(df)):
                end = position + horizon

                if end >= len(df):
                    max_favorable.append(np.nan)
                    max_adverse.append(np.nan)
                    max_drawdown.append(np.nan)
                    continue

                future = values[position:end + 1]

                if len(future) <= 1 or not np.isfinite(future).all():
                    max_favorable.append(np.nan)
                    max_adverse.append(np.nan)
                    max_drawdown.append(np.nan)
                    continue

                entry = future[0]
                returns = future / entry - 1.0

                max_favorable.append(float(np.max(returns) * 100.0))
                max_adverse.append(float(np.min(returns) * 100.0))

                running_max = np.maximum.accumulate(future)
                drawdowns = future / running_max - 1.0
                max_drawdown.append(float(np.min(drawdowns) * 100.0))

            df[f"Max_Favorable_{horizon}W"] = max_favorable
            df[f"Max_Adverse_{horizon}W"] = max_adverse
            df[f"Max_Drawdown_{horizon}W"] = max_drawdown

        return df

    def summarize(self, outcomes: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for horizon in self.horizons:
            drawdown_column = f"Max_Drawdown_{horizon}W"
            favorable_column = f"Max_Favorable_{horizon}W"
            adverse_column = f"Max_Adverse_{horizon}W"

            available = outcomes[
                [drawdown_column, favorable_column, adverse_column]
            ].dropna()

            if available.empty:
                continue

            rows.append(
                {
                    "Horizon": f"{horizon}W",
                    "Count": len(available),
                    "Average_Max_Drawdown": available[
                        drawdown_column
                    ].mean(),
                    "Median_Max_Drawdown": available[
                        drawdown_column
                    ].median(),
                    "Worst_Max_Drawdown": available[
                        drawdown_column
                    ].min(),
                    "Average_Max_Favorable": available[
                        favorable_column
                    ].mean(),
                    "Average_Max_Adverse": available[
                        adverse_column
                    ].mean(),
                }
            )

        return pd.DataFrame(rows)
