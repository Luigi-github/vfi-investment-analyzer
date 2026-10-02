import pandas as pd


class VFIRecoverySignalQuality:

    def __init__(
        self,
        primary_horizon=26,
        secondary_horizon=52
    ):
        self.primary_horizon = primary_horizon
        self.secondary_horizon = secondary_horizon

    def classify_returns(
        self,
        recovery_analysis: pd.DataFrame
    ) -> pd.DataFrame:

        if recovery_analysis.empty:
            return pd.DataFrame()

        df = recovery_analysis.copy()

        primary_column = (
            f"Return_{self.primary_horizon}W"
        )

        secondary_column = (
            f"Return_{self.secondary_horizon}W"
        )

        if primary_column not in df.columns:
            raise ValueError(
                f"Missing required column: {primary_column}"
            )

        df["Primary_Return"] = df[primary_column]

        if secondary_column in df.columns:
            df["Secondary_Return"] = df[
                secondary_column
            ]
        else:
            df["Secondary_Return"] = pd.NA

        df["Primary_Return_Result"] = "NO_DATA"

        df.loc[
            df["Primary_Return"] > 0,
            "Primary_Return_Result"
        ] = "POSITIVE"

        df.loc[
            df["Primary_Return"] <= 0,
            "Primary_Return_Result"
        ] = "NEGATIVE"

        df["Secondary_Return_Result"] = "NO_DATA"

        df.loc[
            df["Secondary_Return"] > 0,
            "Secondary_Return_Result"
        ] = "POSITIVE"

        df.loc[
            df["Secondary_Return"] <= 0,
            "Secondary_Return_Result"
        ] = "NEGATIVE"

        df["Investment_Result"] = "NO_DATA"

        primary_positive = (
            df["Primary_Return"] > 0
        )

        secondary_positive = (
            df["Secondary_Return"] > 0
        )

        secondary_available = (
            df["Secondary_Return"].notna()
        )

        df.loc[
            primary_positive,
            "Investment_Result"
        ] = "POSITIVE_PRIMARY"

        df.loc[
            ~primary_positive,
            "Investment_Result"
        ] = "NEGATIVE_PRIMARY"

        df.loc[
            primary_positive
            & secondary_positive,
            "Investment_Result"
        ] = "POSITIVE_BOTH"

        df.loc[
            primary_positive
            & secondary_available
            & ~secondary_positive,
            "Investment_Result"
        ] = "POSITIVE_PRIMARY_NEGATIVE_SECONDARY"

        df.loc[
            ~primary_positive
            & secondary_positive,
            "Investment_Result"
        ] = "NEGATIVE_PRIMARY_POSITIVE_SECONDARY"

        return df

    def calculate_outcome_summary(
        self,
        classified_data: pd.DataFrame
    ) -> pd.DataFrame:

        if classified_data.empty:
            return pd.DataFrame()

        rows = []

        groups = [
            "CONFIRMED",
            "FAILED"
        ]

        for outcome in groups:

            subset = classified_data.loc[
                classified_data["Recovery_Outcome"]
                == outcome
            ]

            if subset.empty:
                continue

            primary = subset[
                "Primary_Return"
            ].dropna()

            secondary = subset[
                "Secondary_Return"
            ].dropna()

            rows.append({
                "Recovery_Outcome": outcome,
                "Count": len(subset),

                "Primary_Count": len(primary),
                "Primary_Average": (
                    primary.mean()
                    if not primary.empty
                    else None
                ),
                "Primary_Median": (
                    primary.median()
                    if not primary.empty
                    else None
                ),
                "Primary_Positive_Rate": (
                    primary.gt(0).mean() * 100
                    if not primary.empty
                    else None
                ),

                "Secondary_Count": len(secondary),
                "Secondary_Average": (
                    secondary.mean()
                    if not secondary.empty
                    else None
                ),
                "Secondary_Median": (
                    secondary.median()
                    if not secondary.empty
                    else None
                ),
                "Secondary_Positive_Rate": (
                    secondary.gt(0).mean() * 100
                    if not secondary.empty
                    else None
                )
            })

        return pd.DataFrame(rows)

    def calculate_feature_by_return_result(
        self,
        classified_data: pd.DataFrame
    ) -> pd.DataFrame:

        if classified_data.empty:
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

        positive = classified_data.loc[
            classified_data["Primary_Return"] > 0
        ]

        negative = classified_data.loc[
            classified_data["Primary_Return"] <= 0
        ]

        for feature in features:

            if feature not in classified_data.columns:
                continue

            positive_values = positive[
                feature
            ].dropna()

            negative_values = negative[
                feature
            ].dropna()

            if (
                positive_values.empty
                or negative_values.empty
            ):
                continue

            rows.append({
                "Feature": feature,

                "Positive_Count": len(
                    positive_values
                ),
                "Positive_Mean": (
                    positive_values.mean()
                ),
                "Positive_Median": (
                    positive_values.median()
                ),

                "Negative_Count": len(
                    negative_values
                ),
                "Negative_Mean": (
                    negative_values.mean()
                ),
                "Negative_Median": (
                    negative_values.median()
                ),

                "Mean_Difference": (
                    positive_values.mean()
                    - negative_values.mean()
                )
            })

        return pd.DataFrame(rows)

    def calculate_signal_matrix(
        self,
        classified_data: pd.DataFrame
    ) -> pd.DataFrame:

        if classified_data.empty:
            return pd.DataFrame()

        columns = [
            "Date",
            "VFI",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Distance_Zero",
            "VFI_Range_Position",
            "Recovery_Outcome",
            "Primary_Return",
            "Secondary_Return",
            "Primary_Return_Result",
            "Secondary_Return_Result",
            "Investment_Result"
        ]

        available_columns = [
            column
            for column in columns
            if column in classified_data.columns
        ]

        result = classified_data[
            available_columns
        ].copy()

        return result.sort_values(
            by="Date"
        ).reset_index(drop=True)