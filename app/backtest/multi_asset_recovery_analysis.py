import numpy as np
import pandas as pd


class MultiAssetRecoveryAnalyzer:

    def __init__(
        self,
        horizons=(4, 8, 12, 26, 52),
        primary_horizon=26,
        positive_return_threshold=0.0
    ):
        self.horizons = horizons
        self.primary_horizon = primary_horizon
        self.positive_return_threshold = positive_return_threshold

    def validate_data(self, data):
        required_columns = [
            "Close",
            "VFI",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Distance_Zero",
            "VFI_Range_Position",
            "VFI_Cycle_Stage",
            "Recovery_Cluster_ID",
        ]

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    def calculate_forward_returns(self, data):
        df = data.copy()

        if "Close" not in df.columns:
            raise ValueError(
                "Column 'Close' is required."
            )

        for horizon in self.horizons:

            future_price = df["Close"].shift(
                -horizon
            )

            df[f"Return_{horizon}W"] = (
                (
                    future_price
                    / df["Close"]
                )
                - 1.0
            ) * 100.0

        return df

    def select_causal_cluster_representatives(self, data):
        """Select the first observed event in each recovery cluster."""
        if data.empty:
            return data.copy()

        if "Recovery_Cluster_ID" not in data.columns:
            raise ValueError(
                "Column 'Recovery_Cluster_ID' is required."
            )

        representatives = []

        for cluster_id, group in data.groupby(
            "Recovery_Cluster_ID"
        ):
            group = group.sort_index()
            representative = group.iloc[0].copy()
            representative["Recovery_Cluster_ID"] = cluster_id
            representative["Cluster_Start_Date"] = group.index[0]
            representative["Representative_Method"] = (
                "CAUSAL_FIRST_RECOVERY"
            )
            representative["Representative_Causal"] = True
            representatives.append(representative)

        if not representatives:
            return pd.DataFrame()

        result = pd.DataFrame(representatives)
        result.index.name = data.index.name
        return result.sort_index()

    def select_cluster_representatives(self, data):
        if data.empty:
            return data.copy()

        if "Recovery_Cluster_ID" not in data.columns:
            raise ValueError(
                "Column 'Recovery_Cluster_ID' is required."
            )

        representatives = []

        for cluster_id, group in data.groupby(
            "Recovery_Cluster_ID"
        ):

            group = group.sort_index()

            # Por ahora mantenemos la misma regla
            # utilizada en DDD:
            # representante = menor VFI del cluster.
            representative_index = (
                group["VFI"].idxmin()
            )

            representative = (
                group.loc[representative_index]
                .copy()
            )

            representative[
                "Recovery_Cluster_ID"
            ] = cluster_id

            representative[
                "Cluster_Start_Date"
            ] = group.index.min()

            representative[
                "Cluster_End_Date"
            ] = group.index.max()

            representative[
                "Cluster_Size"
            ] = len(group)

            representatives.append(
                representative
            )

        if not representatives:
            return pd.DataFrame()

        result = pd.DataFrame(
            representatives
        )

        result.index.name = data.index.name

        return result.sort_index()

    def classify_results(self, data):
        df = data.copy()

        primary_column = (
            f"Return_{self.primary_horizon}W"
        )

        if primary_column not in df.columns:
            raise ValueError(
                f"Column '{primary_column}' is required."
            )

        df["Primary_Return"] = (
            df[primary_column]
        )

        df["Primary_Result"] = np.where(
            df["Primary_Return"].isna(),
            "INSUFFICIENT_DATA",
            np.where(
                df["Primary_Return"]
                > self.positive_return_threshold,
                "POSITIVE",
                "NEGATIVE"
            )
        )

        if "Return_52W" in df.columns:

            df["Long_Term_Result"] = np.where(
                df["Return_52W"].isna(),
                "INSUFFICIENT_DATA",
                np.where(
                    df["Return_52W"]
                    > self.positive_return_threshold,
                    "POSITIVE",
                    "NEGATIVE"
                )
            )

        return df

    def analyze_asset_causal(
        self,
        symbol,
        data
    ):
        """Analyze one asset using point-in-time representatives."""
        self.validate_data(data)
        df = self.calculate_forward_returns(data.copy())
        representatives = self.select_causal_cluster_representatives(df)

        if representatives.empty:
            return representatives

        representatives = self.classify_results(representatives)
        representatives["Symbol"] = symbol
        return representatives

    def analyze_asset(
        self,
        symbol,
        data
    ):
        self.validate_data(data)

        df = data.copy()

        df = self.calculate_forward_returns(
            df
        )

        representatives = (
            self.select_cluster_representatives(
                df
            )
        )

        if representatives.empty:
            return representatives

        representatives = (
            self.classify_results(
                representatives
            )
        )

        representatives[
            "Symbol"
        ] = symbol

        return representatives

    def combine_assets(
        self,
        asset_results
    ):
        valid_results = [
            result
            for result in asset_results
            if result is not None
            and not result.empty
        ]

        if not valid_results:
            return pd.DataFrame()

        combined = pd.concat(
            valid_results,
            axis=0
        )

        combined = combined.sort_values(
            ["Symbol", combined.index.name]
        )

        return combined

    def asset_summary(
        self,
        combined_data
    ):
        if combined_data.empty:
            return pd.DataFrame()

        rows = []

        for symbol, group in combined_data.groupby(
            "Symbol"
        ):

            valid = group[
                group["Primary_Result"].isin(
                    ["POSITIVE", "NEGATIVE"]
                )
            ]

            positive_count = (
                valid["Primary_Result"]
                .eq("POSITIVE")
                .sum()
            )

            negative_count = (
                valid["Primary_Result"]
                .eq("NEGATIVE")
                .sum()
            )

            total = (
                positive_count
                + negative_count
            )

            positive_rate = (
                positive_count / total * 100
                if total > 0
                else np.nan
            )

            rows.append({
                "Symbol": symbol,
                "Clusters": len(group),
                "Valid_Clusters": total,
                "Positive_26W": positive_count,
                "Negative_26W": negative_count,
                "Positive_Rate_26W":
                    positive_rate,
                "Average_Return_26W":
                    valid[
                        "Return_26W"
                    ].mean(),
                "Median_Return_26W":
                    valid[
                        "Return_26W"
                    ].median(),
                "Average_Return_52W":
                    valid[
                        "Return_52W"
                    ].mean()
                    if "Return_52W"
                    in valid.columns
                    else np.nan,
            })

        return pd.DataFrame(rows)

    def global_summary(
        self,
        combined_data
    ):
        if combined_data.empty:
            return pd.DataFrame()

        valid = combined_data[
            combined_data["Primary_Result"].isin(
                ["POSITIVE", "NEGATIVE"]
            )
        ]

        if valid.empty:
            return pd.DataFrame()

        positive_count = (
            valid["Primary_Result"]
            .eq("POSITIVE")
            .sum()
        )

        negative_count = (
            valid["Primary_Result"]
            .eq("NEGATIVE")
            .sum()
        )

        total = (
            positive_count
            + negative_count
        )

        return pd.DataFrame([{
            "Assets": valid["Symbol"].nunique(),
            "Total_Clusters": len(
                combined_data
            ),
            "Valid_Clusters": total,
            "Positive_26W": positive_count,
            "Negative_26W": negative_count,
            "Positive_Rate_26W":
                positive_count
                / total
                * 100,
            "Average_Return_26W":
                valid[
                    "Return_26W"
                ].mean(),
            "Median_Return_26W":
                valid[
                    "Return_26W"
                ].median(),
            "Average_Return_52W":
                valid[
                    "Return_52W"
                ].mean()
                if "Return_52W"
                in valid.columns
                else np.nan,
        }])

    def condition_analysis(
        self,
        combined_data
    ):
        if combined_data.empty:
            return pd.DataFrame()

        valid = combined_data[
            combined_data["Primary_Result"].isin(
                ["POSITIVE", "NEGATIVE"]
            )
        ].copy()

        conditions = {

            "VFI < 0":
                valid["VFI"] < 0,

            "VFI < -5":
                valid["VFI"] < -5,

            "VFI near zero":
                valid["VFI_Distance_Zero"] <= 1,

            "Slope > 0":
                valid["VFI_Slope"] > 0,

            "Slope 4W > 0":
                valid["VFI_Slope_4W"] > 0,

            "Slope 8W > 0":
                valid["VFI_Slope_8W"] > 0,

            "Range position <= 20%":
                valid["VFI_Range_Position"] <= 0.20,

            "Slope > 0 AND Slope 8W > 0":
                (
                    (valid["VFI_Slope"] > 0)
                    &
                    (valid["VFI_Slope_8W"] > 0)
                ),

            "Slope 4W > 0 AND Slope 8W > 0":
                (
                    (valid["VFI_Slope_4W"] > 0)
                    &
                    (valid["VFI_Slope_8W"] > 0)
                ),

            "VFI < 0 AND Slope > 0":
                (
                    (valid["VFI"] < 0)
                    &
                    (valid["VFI_Slope"] > 0)
                ),

            "VFI < -5 AND Slope > 0":
                (
                    (valid["VFI"] < -5)
                    &
                    (valid["VFI_Slope"] > 0)
                ),

            "VFI < 0 AND Slope > 0 AND Slope 8W > 0":
                (
                    (valid["VFI"] < 0)
                    &
                    (valid["VFI_Slope"] > 0)
                    &
                    (valid["VFI_Slope_8W"] > 0)
                ),
        }

        rows = []

        for condition_name, mask in conditions.items():

            selected = valid[mask]

            if selected.empty:

                rows.append({
                    "Condition": condition_name,
                    "Clusters": 0,
                    "Assets": 0,
                    "Positive_26W": 0,
                    "Negative_26W": 0,
                    "Positive_Rate_26W": np.nan,
                    "Average_Return_26W": np.nan,
                    "Median_Return_26W": np.nan,
                    "Average_Return_52W": np.nan,
                })

                continue

            positive_count = (
                selected["Primary_Result"]
                .eq("POSITIVE")
                .sum()
            )

            negative_count = (
                selected["Primary_Result"]
                .eq("NEGATIVE")
                .sum()
            )

            total = (
                positive_count
                + negative_count
            )

            rows.append({
                "Condition": condition_name,
                "Clusters": len(selected),
                "Assets": selected[
                    "Symbol"
                ].nunique(),
                "Positive_26W": positive_count,
                "Negative_26W": negative_count,
                "Positive_Rate_26W":
                    positive_count
                    / total
                    * 100
                    if total > 0
                    else np.nan,
                "Average_Return_26W":
                    selected[
                        "Return_26W"
                    ].mean(),
                "Median_Return_26W":
                    selected[
                        "Return_26W"
                    ].median(),
                "Average_Return_52W":
                    selected[
                        "Return_52W"
                    ].mean()
                    if "Return_52W"
                    in selected.columns
                    else np.nan,
            })

        return pd.DataFrame(rows)

    def analyze_causal(
        self,
        asset_data
    ):
        """Run the multi-asset analysis with causal representatives."""
        asset_results = []

        for symbol, data in asset_data.items():
            result = self.analyze_asset_causal(symbol, data)
            if not result.empty:
                asset_results.append(result)

        combined = self.combine_assets(asset_results)

        return {
            "asset_results": asset_results,
            "combined": combined,
            "asset_summary": self.asset_summary(combined),
            "global_summary": self.global_summary(combined),
            "condition_analysis": self.condition_analysis(combined),
        }

    def analyze(
        self,
        asset_data
    ):
        """
        asset_data debe tener esta estructura:

        {
            "DDD": dataframe,
            "INTC": dataframe,
            "NVDA": dataframe,
            "BTC-USD": dataframe
        }
        """

        asset_results = []

        for symbol, data in asset_data.items():

            result = self.analyze_asset(
                symbol,
                data
            )

            if not result.empty:
                asset_results.append(
                    result
                )

        combined = self.combine_assets(
            asset_results
        )

        return {
            "asset_results": asset_results,
            "combined": combined,
            "asset_summary":
                self.asset_summary(
                    combined
                ),
            "global_summary":
                self.global_summary(
                    combined
                ),
            "condition_analysis":
                self.condition_analysis(
                    combined
                ),
        }