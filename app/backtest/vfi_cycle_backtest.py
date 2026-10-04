import pandas as pd

from app.backtest.execution_costs import LongOnlyExecutionCostModel


class VFICycleBacktest:

    def __init__(
        self,
        horizons=(4, 8, 12, 26, 52),
        execution_cost_model=None
    ):
        self.horizons = tuple(horizons)

        if execution_cost_model is not None and not isinstance(
            execution_cost_model, LongOnlyExecutionCostModel
        ):
            raise TypeError(
                "execution_cost_model must be a LongOnlyExecutionCostModel or None."
            )

        self.execution_cost_model = execution_cost_model

    def detect_transitions(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        df = data.copy()

        required_columns = [
            "VFI",
            "VFI_Cycle_Stage",
            "Cycle_Transition",
            "Cycle_Stage_Changed",
            "Weeks_In_Current_Stage",
            "Recovery_Phase",
            "Recovery_Quality",
            "Accumulation_Phase",
            "Accumulation_Confirmed",
            "Zero_Cross_Up_Event",
            "Zero_Cross_Down_Event"
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        transitions = df[
            df["Cycle_Stage_Changed"]
        ].copy()

        transitions["Transition_Type"] = "OTHER"

        recovery_transitions = transitions[
            "VFI_Cycle_Stage"
        ].isin([
            "EARLY_RECOVERY",
            "SUSTAINED_RECOVERY",
            "APPROACHING_ZERO",
            "ZERO_CROSS_UP",
            "ACCUMULATION"
        ])

        transitions.loc[
            recovery_transitions,
            "Transition_Type"
        ] = "RECOVERY"

        accumulation_transitions = transitions[
            "Accumulation_Phase"
        ]

        transitions.loc[
            accumulation_transitions,
            "Transition_Type"
        ] = "ACCUMULATION"

        bearish_transitions = transitions[
            "VFI_Cycle_Stage"
        ] == "BEARISH_FLOW"

        transitions.loc[
            bearish_transitions,
            "Transition_Type"
        ] = "BEARISH"

        transitions["Zero_Cross"] = "NONE"

        transitions.loc[
            transitions["Zero_Cross_Up_Event"],
            "Zero_Cross"
        ] = "UP"

        transitions.loc[
            transitions["Zero_Cross_Down_Event"],
            "Zero_Cross"
        ] = "DOWN"

        transitions["Recovery_Signal"] = (
            transitions["Recovery_Phase"]
        )

        transitions["Accumulation_Signal"] = (
            transitions["Accumulation_Phase"]
            |
            transitions["Accumulation_Confirmed"]
        )

        return transitions

    def calculate_stage_summary(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        df = data.copy()

        required_columns = [
            "VFI_Cycle_Stage",
            "VFI",
            "Weeks_In_Current_Stage"
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        summary = (
            df
            .groupby("VFI_Cycle_Stage")
            .agg(
                Weeks=(
                    "Weeks_In_Current_Stage",
                    "count"
                ),
                Average_VFI=(
                    "VFI",
                    "mean"
                ),
                Minimum_VFI=(
                    "VFI",
                    "min"
                ),
                Maximum_VFI=(
                    "VFI",
                    "max"
                )
            )
            .reset_index()
        )

        return summary

    def calculate_transition_summary(
        self,
        transitions: pd.DataFrame
    ) -> pd.DataFrame:

        if transitions.empty:
            return pd.DataFrame(
                columns=[
                    "Transition_Type",
                    "Count"
                ]
            )

        summary = (
            transitions
            .groupby("Transition_Type")
            .size()
            .reset_index(name="Count")
        )

        return summary

    def calculate_recovery_summary(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        df = data.copy()

        recovery = df[
            df["Recovery_Phase"]
        ].copy()

        if recovery.empty:
            return pd.DataFrame(
                columns=[
                    "Recovery_Quality",
                    "Count"
                ]
            )

        summary = (
            recovery
            .groupby("Recovery_Quality")
            .size()
            .reset_index(name="Count")
        )

        return summary

    def calculate_forward_returns(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        df = data.copy()

        if "Close" not in df.columns:
            raise ValueError(
                "Missing required column: Close"
            )

        for horizon in self.horizons:

            future_price = (
                df["Close"]
                .shift(-horizon)
            )

            gross_return = (
                (
                    future_price
                    / df["Close"]
                )
                - 1.0
            ) * 100.0

            df[f"Return_{horizon}W"] = gross_return

            if self.execution_cost_model is not None:
                df[f"Net_Return_{horizon}W"] = (
                    self.execution_cost_model.net_return(gross_return)
                )

        return df

    def calculate_transition_returns(
        self,
        data: pd.DataFrame,
        transitions: pd.DataFrame
    ) -> pd.DataFrame:

        df = self.calculate_forward_returns(
            data
        )

        transition_dates = transitions.index

        rows = []

        for date in transition_dates:

            if date not in df.index:
                continue

            row = df.loc[date]

            result = {
                "Date": date,
                "VFI_Cycle_Stage": row[
                    "VFI_Cycle_Stage"
                ],
                "Cycle_Transition": row[
                    "Cycle_Transition"
                ],
                "VFI": row["VFI"],
                "Close": row["Close"]
            }

            for horizon in self.horizons:

                column = f"Return_{horizon}W"

                result[column] = row[column]

            rows.append(result)

        if not rows:

            columns = [
                "Date",
                "VFI_Cycle_Stage",
                "Cycle_Transition",
                "VFI",
                "Close"
            ]

            columns += [
                f"Return_{horizon}W"
                for horizon in self.horizons
            ]

            return pd.DataFrame(
                columns=columns
            )

        return pd.DataFrame(rows)

    def calculate_transition_performance(
        self,
        transition_returns: pd.DataFrame
    ) -> pd.DataFrame:

        if transition_returns.empty:

            columns = [
                "VFI_Cycle_Stage",
                "Transitions"
            ]

            columns += [
                f"Average_Return_{horizon}W"
                for horizon in self.horizons
            ]

            return pd.DataFrame(
                columns=columns
            )

        aggregation = {
            "Date": "count"
        }

        for horizon in self.horizons:

            aggregation[
                f"Return_{horizon}W"
            ] = "mean"

        summary = (
            transition_returns
            .groupby("VFI_Cycle_Stage")
            .agg(aggregation)
            .reset_index()
        )

        summary = summary.rename(
            columns={
                "Date": "Transitions"
            }
        )

        rename_map = {}

        for horizon in self.horizons:

            rename_map[
                f"Return_{horizon}W"
            ] = (
                f"Average_Return_{horizon}W"
            )

        summary = summary.rename(
            columns=rename_map
        )

        return summary

    def calculate_stage_forward_returns(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        df = self.calculate_forward_returns(
            data
        )

        required_columns = [
            "VFI_Cycle_Stage",
            "Close"
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        rows = []

        for stage, group in df.groupby(
            "VFI_Cycle_Stage"
        ):

            result = {
                "VFI_Cycle_Stage": stage,
                "Weeks": len(group)
            }

            for horizon in self.horizons:

                column = f"Return_{horizon}W"

                result[
                    f"Average_Return_{horizon}W"
                ] = group[column].mean()

                result[
                    f"Median_Return_{horizon}W"
                ] = group[column].median()

                result[
                    f"Positive_Return_Rate_{horizon}W"
                ] = (
                    group[column]
                    .dropna()
                    .gt(0)
                    .mean()
                    * 100
                )

            rows.append(result)

        return pd.DataFrame(rows)