import pandas as pd


class VFIRecoveryCycleQuality:

    def __init__(
        self,
        horizons=(4, 8, 12, 26, 52)
    ):
        self.horizons = horizons

    def analyze(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        if data.empty:
            return pd.DataFrame()

        df = data.copy()

        recovery_stages = [
            "EARLY_RECOVERY",
            "SUSTAINED_RECOVERY",
            "APPROACHING_ZERO",
            "ZERO_CROSS_UP",
            "ACCUMULATION"
        ]

        df = df[
            df["VFI_Cycle_Stage"].isin(
                recovery_stages
            )
        ].copy()

        if df.empty:
            return pd.DataFrame()

        return df.reset_index()

    def calculate_stage_summary(
        self,
        recovery_data: pd.DataFrame
    ) -> pd.DataFrame:

        if recovery_data.empty:
            return pd.DataFrame()

        rows = []

        for stage, group in recovery_data.groupby(
            "VFI_Cycle_Stage"
        ):

            row = {
                "Stage": stage,
                "Count": len(group)
            }

            for horizon in self.horizons:

                column = f"Return_{horizon}W"

                if column not in group.columns:
                    continue

                values = group[column].dropna()

                if values.empty:
                    row[
                        f"Count_{horizon}W"
                    ] = 0

                    row[
                        f"Average_Return_{horizon}W"
                    ] = None

                    row[
                        f"Median_Return_{horizon}W"
                    ] = None

                    row[
                        f"Positive_Rate_{horizon}W"
                    ] = None

                    continue

                row[
                    f"Count_{horizon}W"
                ] = len(values)

                row[
                    f"Average_Return_{horizon}W"
                ] = values.mean()

                row[
                    f"Median_Return_{horizon}W"
                ] = values.median()

                row[
                    f"Positive_Rate_{horizon}W"
                ] = (
                    values.gt(0).mean()
                    * 100
                )

            rows.append(row)

        return pd.DataFrame(rows)

    def calculate_feature_summary(
        self,
        recovery_data: pd.DataFrame
    ) -> pd.DataFrame:

        if recovery_data.empty:
            return pd.DataFrame()

        features = [
            "VFI",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Distance_Zero",
            "VFI_Range_Position"
        ]

        rows = []

        for stage, group in recovery_data.groupby(
            "VFI_Cycle_Stage"
        ):

            for feature in features:

                if feature not in group.columns:
                    continue

                values = group[
                    feature
                ].dropna()

                if values.empty:
                    continue

                rows.append({
                    "Stage": stage,
                    "Feature": feature,
                    "Count": len(values),
                    "Mean": values.mean(),
                    "Median": values.median(),
                    "Minimum": values.min(),
                    "Maximum": values.max()
                })

        return pd.DataFrame(rows)

    def calculate_transition_quality(
        self,
        recovery_data: pd.DataFrame
    ) -> pd.DataFrame:

        if recovery_data.empty:
            return pd.DataFrame()

        rows = []

        for stage, group in recovery_data.groupby(
            "VFI_Cycle_Stage"
        ):

            row = {
                "Stage": stage,
                "Count": len(group)
            }

            if "Cycle_Stage_Changed" in group.columns:
                row["Stage_Changes"] = (
                    group["Cycle_Stage_Changed"]
                    .sum()
                )

            if "Recovery_Phase" in group.columns:
                row["Recovery_Phase_Count"] = (
                    group["Recovery_Phase"]
                    .sum()
                )

            if "Accumulation_Phase" in group.columns:
                row["Accumulation_Phase_Count"] = (
                    group["Accumulation_Phase"]
                    .sum()
                )

            if "Accumulation_Confirmed" in group.columns:
                row["Accumulation_Confirmed_Count"] = (
                    group["Accumulation_Confirmed"]
                    .sum()
                )

            rows.append(row)

        return pd.DataFrame(rows)