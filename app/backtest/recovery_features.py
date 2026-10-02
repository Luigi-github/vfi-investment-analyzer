import pandas as pd


class VFIRecoveryFeatures:

    def __init__(
        self,
        horizons=(4, 8, 12, 26, 52)
    ):
        self.horizons = horizons

    def calculate_feature_summary(
        self,
        recovery_analysis: pd.DataFrame
    ) -> pd.DataFrame:

        if recovery_analysis.empty:
            return pd.DataFrame()

        features = [
            "VFI",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Distance_Zero",
            "VFI_Range_Position",
        ]

        rows = []

        for feature in features:

            if feature not in recovery_analysis.columns:
                continue

            confirmed = recovery_analysis.loc[
                recovery_analysis["Recovery_Outcome"] == "CONFIRMED",
                feature
            ].dropna()

            failed = recovery_analysis.loc[
                recovery_analysis["Recovery_Outcome"] == "FAILED",
                feature
            ].dropna()

            row = {
                "Feature": feature,

                "Confirmed_Count": len(confirmed),
                "Confirmed_Mean": confirmed.mean(),
                "Confirmed_Median": confirmed.median(),
                "Confirmed_Min": confirmed.min(),
                "Confirmed_Max": confirmed.max(),

                "Failed_Count": len(failed),
                "Failed_Mean": failed.mean(),
                "Failed_Median": failed.median(),
                "Failed_Min": failed.min(),
                "Failed_Max": failed.max(),
            }

            if (
                not confirmed.empty
                and not failed.empty
            ):
                row["Mean_Difference"] = (
                    confirmed.mean()
                    - failed.mean()
                )
            else:
                row["Mean_Difference"] = None

            rows.append(row)

        return pd.DataFrame(rows)

    def calculate_return_summary(
        self,
        recovery_analysis: pd.DataFrame
    ) -> pd.DataFrame:

        if recovery_analysis.empty:
            return pd.DataFrame()

        rows = []

        for horizon in self.horizons:

            column = f"Return_{horizon}W"

            if column not in recovery_analysis.columns:
                continue

            for outcome in [
                "CONFIRMED",
                "FAILED"
            ]:

                values = recovery_analysis.loc[
                    recovery_analysis["Recovery_Outcome"]
                    == outcome,
                    column
                ].dropna()

                if values.empty:
                    continue

                rows.append({
                    "Horizon": f"{horizon}W",
                    "Recovery_Outcome": outcome,
                    "Count": len(values),
                    "Average_Return": values.mean(),
                    "Median_Return": values.median(),
                    "Minimum_Return": values.min(),
                    "Maximum_Return": values.max(),
                    "Positive_Return_Rate": (
                        values.gt(0).mean() * 100
                    )
                })

        return pd.DataFrame(rows)

    def calculate_feature_ranges(
        self,
        recovery_analysis: pd.DataFrame
    ) -> pd.DataFrame:

        if recovery_analysis.empty:
            return pd.DataFrame()

        features = [
            "VFI",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Distance_Zero",
            "VFI_Range_Position",
        ]

        rows = []

        for feature in features:

            if feature not in recovery_analysis.columns:
                continue

            confirmed = recovery_analysis.loc[
                recovery_analysis["Recovery_Outcome"] == "CONFIRMED",
                feature
            ].dropna()

            failed = recovery_analysis.loc[
                recovery_analysis["Recovery_Outcome"] == "FAILED",
                feature
            ].dropna()

            if confirmed.empty or failed.empty:
                continue

            rows.append({
                "Feature": feature,

                "Confirmed_Min": confirmed.min(),
                "Confirmed_Max": confirmed.max(),

                "Failed_Min": failed.min(),
                "Failed_Max": failed.max(),

                "Confirmed_Median": confirmed.median(),
                "Failed_Median": failed.median()
            })

        return pd.DataFrame(rows)