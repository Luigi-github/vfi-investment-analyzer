import pandas as pd


class VFIRecoveryAnalysis:

    def __init__(
        self,
        forward_horizons=(4, 8, 12, 26, 52)
    ):
        self.forward_horizons = forward_horizons

    def analyze_early_recovery(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        df = data.copy()

        required_columns = [
            "VFI",
            "Close",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Distance_Zero",
            "VFI_Cycle_Stage",
            "Cycle_Transition",
            "VFI_Range_Position",
            "Accumulation_Confirmed"
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

        df = self._calculate_forward_returns(df)

        early_recovery_dates = df.index[
            df["VFI_Cycle_Stage"] == "EARLY_RECOVERY"
        ]

        rows = []

        for date in early_recovery_dates:

            position = df.index.get_loc(date)

            current_row = df.loc[date]

            future_data = df.iloc[position + 1:]

            if future_data.empty:
                continue

            future_stage = self._find_next_relevant_stage(
                future_data
            )

            if future_stage is None:
                outcome = "UNRESOLVED"

            elif future_stage in [
                "SUSTAINED_RECOVERY",
                "APPROACHING_ZERO",
                "ZERO_CROSS_UP",
                "ACCUMULATION"
            ]:
                outcome = "CONFIRMED"

            elif future_stage in [
                "BEARISH_FLOW",
                "NEUTRAL",
                "ZERO_CROSS_DOWN"
            ]:
                outcome = "FAILED"

            else:
                outcome = "UNRESOLVED"

            result = {
                "Date": date,
                "Close": current_row["Close"],
                "VFI": current_row["VFI"],
                "VFI_Slope": current_row["VFI_Slope"],
                "VFI_Slope_4W": current_row["VFI_Slope_4W"],
                "VFI_Slope_8W": current_row["VFI_Slope_8W"],
                "VFI_Distance_Zero": (
                    current_row["VFI_Distance_Zero"]
                ),
                "VFI_Range_Position": (
                    current_row["VFI_Range_Position"]
                ),
                "Accumulation_Confirmed": (
                    current_row["Accumulation_Confirmed"]
                ),
                "Next_Relevant_Stage": future_stage,
                "Recovery_Outcome": outcome
            }

            for horizon in self.forward_horizons:

                column = f"Return_{horizon}W"

                result[column] = current_row[column]

            rows.append(result)

        if not rows:

            return pd.DataFrame(
                columns=[
                    "Date",
                    "Close",
                    "VFI",
                    "VFI_Slope",
                    "VFI_Slope_4W",
                    "VFI_Slope_8W",
                    "VFI_Distance_Zero",
                    "VFI_Range_Position",
                    "Accumulation_Confirmed",
                    "Next_Relevant_Stage",
                    "Recovery_Outcome"
                ]
                + [
                    f"Return_{horizon}W"
                    for horizon in self.forward_horizons
                ]
            )

        return pd.DataFrame(rows)

    def _find_next_relevant_stage(
        self,
        future_data: pd.DataFrame
    ):

        relevant_stages = [
            "SUSTAINED_RECOVERY",
            "APPROACHING_ZERO",
            "ZERO_CROSS_UP",
            "ACCUMULATION",
            "BEARISH_FLOW",
            "NEUTRAL",
            "ZERO_CROSS_DOWN"
        ]

        for stage in future_data[
            "VFI_Cycle_Stage"
        ]:

            if stage in relevant_stages:
                return stage

        return None

    def _calculate_forward_returns(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        df = data.copy()

        for horizon in self.forward_horizons:

            future_price = (
                df["Close"].shift(-horizon)
            )

            df[
                f"Return_{horizon}W"
            ] = (
                (
                    future_price
                    / df["Close"]
                )
                - 1.0
            ) * 100.0

        return df

    def calculate_outcome_summary(
        self,
        recovery_analysis: pd.DataFrame
    ) -> pd.DataFrame:

        if recovery_analysis.empty:

            return pd.DataFrame(
                columns=[
                    "Recovery_Outcome",
                    "Count"
                ]
            )

        summary = (
            recovery_analysis
            .groupby("Recovery_Outcome")
            .size()
            .reset_index(name="Count")
        )

        return summary

    def calculate_outcome_performance(
        self,
        recovery_analysis: pd.DataFrame
    ) -> pd.DataFrame:

        if recovery_analysis.empty:

            columns = [
                "Recovery_Outcome",
                "Count"
            ]

            columns += [
                f"Average_Return_{horizon}W"
                for horizon in self.forward_horizons
            ]

            columns += [
                f"Median_Return_{horizon}W"
                for horizon in self.forward_horizons
            ]

            return pd.DataFrame(
                columns=columns
            )

        aggregation = {
            "Date": "count"
        }

        for horizon in self.forward_horizons:

            aggregation[
                f"Return_{horizon}W"
            ] = ["mean", "median"]

        summary = (
            recovery_analysis
            .groupby("Recovery_Outcome")
            .agg(aggregation)
        )

        summary.columns = [
            "_".join(column).strip("_")
            for column in summary.columns
        ]

        summary = summary.reset_index()

        rename_map = {
            "Date_count": "Count"
        }

        for horizon in self.forward_horizons:

            rename_map[
                f"Return_{horizon}W_mean"
            ] = (
                f"Average_Return_{horizon}W"
            )

            rename_map[
                f"Return_{horizon}W_median"
            ] = (
                f"Median_Return_{horizon}W"
            )

        summary = summary.rename(
            columns=rename_map
        )

        return summary