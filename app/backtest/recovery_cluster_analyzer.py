import numpy as np
import pandas as pd


class RecoveryClusterAnalyzer:

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
            "VFI_Slope_8W",
            "VFI_Range_Position",
            "Recovery_Cluster_ID"
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

        df = df.sort_index()

        for horizon in self.horizons:

            future_price = (
                df["Close"].shift(-horizon)
            )

            df[f"Return_{horizon}W"] = (
                (
                    future_price
                    / df["Close"]
                )
                - 1.0
            ) * 100.0

        return df

    def _normalize_series(self, series):

        series = pd.to_numeric(
            series,
            errors="coerce"
        )

        valid = series.dropna()

        if valid.empty:
            return pd.Series(
                np.nan,
                index=series.index
            )

        minimum = valid.min()
        maximum = valid.max()

        if np.isclose(
            maximum,
            minimum
        ):
            return pd.Series(
                0.5,
                index=series.index
            )

        return (
            (series - minimum)
            / (maximum - minimum)
        )

    def select_min_vfi(self, cluster):

        if cluster.empty:
            return None

        valid = cluster[
            cluster["VFI"].notna()
        ]

        if valid.empty:
            return None

        return valid[
            "VFI"
        ].idxmin()

    def select_first_recovery(self, cluster):

        if cluster.empty:
            return None

        cluster = cluster.sort_index()

        return cluster.index[0]

    def select_causal_first_recovery(self, cluster):

        """
        Select the first observed event in a cluster.

        This method is causal only when Recovery_Cluster_ID itself was
        assigned online, using information available at each event date.
        It intentionally does not inspect later VFI values to choose the
        representative.
        """

        return self.select_first_recovery(cluster)

    def select_best_recovery(self, cluster):

        if cluster.empty:
            return None, True

        candidates = cluster.copy()

        if "VFI_Slope" in candidates.columns:

            candidates[
                "_Normalized_Slope"
            ] = self._normalize_series(
                candidates["VFI_Slope"]
            )

        else:

            candidates[
                "_Normalized_Slope"
            ] = np.nan

        if "VFI_Slope_8W" in candidates.columns:

            candidates[
                "_Normalized_Slope_8W"
            ] = self._normalize_series(
                candidates["VFI_Slope_8W"]
            )

        else:

            candidates[
                "_Normalized_Slope_8W"
            ] = np.nan

        if "VFI_Range_Position" in candidates.columns:

            range_position = pd.to_numeric(
                candidates["VFI_Range_Position"],
                errors="coerce"
            )

            candidates[
                "_Lower_Range_Score"
            ] = 1.0 - range_position

        else:

            candidates[
                "_Lower_Range_Score"
            ] = np.nan

        components = [
            (
                "_Normalized_Slope",
                0.40
            ),
            (
                "_Normalized_Slope_8W",
                0.40
            ),
            (
                "_Lower_Range_Score",
                0.20
            )
        ]

        scores = {}

        for index, row in candidates.iterrows():

            numerator = 0.0
            denominator = 0.0

            for column, weight in components:

                value = row.get(
                    column,
                    np.nan
                )

                if pd.notna(value):

                    numerator += (
                        float(value)
                        * weight
                    )

                    denominator += weight

            if denominator > 0:

                scores[index] = (
                    numerator
                    / denominator
                )

            else:

                scores[index] = np.nan

        candidates[
            "_Recovery_Score"
        ] = pd.Series(
            scores,
            dtype=float
        )

        valid_scores = candidates[
            candidates[
                "_Recovery_Score"
            ].notna()
        ]

        if valid_scores.empty:

            representative = (
                self.select_min_vfi(
                    cluster
                )
            )

            return representative, True

        representative = (
            valid_scores[
                "_Recovery_Score"
            ].idxmax()
        )

        return representative, False

    def select_representatives(
        self,
        data,
        method
    ):

        if data.empty:
            return data.copy()

        method = method.upper()

        representatives = []

        for cluster_id, cluster in data.groupby(
            "Recovery_Cluster_ID"
        ):

            cluster = cluster.sort_index()

            representative_index = None
            fallback_used = False

            if method == "MIN_VFI":

                representative_index = (
                    self.select_min_vfi(
                        cluster
                    )
                )

            elif method == "FIRST_RECOVERY":

                representative_index = (
                    self.select_first_recovery(
                        cluster
                    )
                )

            elif method == "CAUSAL_FIRST_RECOVERY":

                representative_index = (
                    self.select_causal_first_recovery(
                        cluster
                    )
                )

            elif method == "BEST_RECOVERY":

                (
                    representative_index,
                    fallback_used
                ) = self.select_best_recovery(
                    cluster
                )

            else:

                raise ValueError(
                    "Unknown representative method: "
                    f"{method}"
                )

            if representative_index is None:
                continue

            representative = (
                data.loc[
                    representative_index
                ]
                .copy()
            )

            representative[
                "Recovery_Cluster_ID"
            ] = cluster_id

            representative[
                "Cluster_Start_Date"
            ] = cluster.index.min()

            representative[
                "Cluster_End_Date"
            ] = cluster.index.max()

            representative[
                "Cluster_Size"
            ] = len(cluster)

            representative[
                "Representative_Method"
            ] = method

            representative[
                "Representative_Causal"
            ] = method == "CAUSAL_FIRST_RECOVERY"

            representative[
                "Fallback_Used"
            ] = fallback_used

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

        for horizon in self.horizons:

            column = (
                f"Return_{horizon}W"
            )

            if column not in df.columns:
                continue

            df[
                f"Result_{horizon}W"
            ] = np.where(
                df[column].isna(),
                "INSUFFICIENT_DATA",
                np.where(
                    df[column]
                    > self.positive_return_threshold,
                    "POSITIVE",
                    "NEGATIVE"
                )
            )

        return df

    def summarize_method(
        self,
        data,
        method
    ):

        if data.empty:

            return {
                "Method": method,
                "Clusters": 0,
                "Valid_Clusters": 0,
                "Positive_26W": 0,
                "Negative_26W": 0,
                "Positive_Rate_26W": np.nan,
                "Average_Return_26W": np.nan,
                "Median_Return_26W": np.nan,
                "Average_Return_52W": np.nan,
                "Median_Return_52W": np.nan,
                "Fallback_Count": 0
            }

        valid = data[
            data["Primary_Result"].isin(
                [
                    "POSITIVE",
                    "NEGATIVE"
                ]
            )
        ].copy()

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
            positive_count
            / total
            * 100.0
            if total > 0
            else np.nan
        )

        average_return_26 = (
            valid["Return_26W"].mean()
            if "Return_26W" in valid.columns
            else np.nan
        )

        median_return_26 = (
            valid["Return_26W"].median()
            if "Return_26W" in valid.columns
            else np.nan
        )

        average_return_52 = (
            valid["Return_52W"].mean()
            if "Return_52W" in valid.columns
            else np.nan
        )

        median_return_52 = (
            valid["Return_52W"].median()
            if "Return_52W" in valid.columns
            else np.nan
        )

        fallback_count = 0

        if "Fallback_Used" in data.columns:

            fallback_count = int(
                data[
                    "Fallback_Used"
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        return {
            "Method": method,
            "Clusters": len(data),
            "Valid_Clusters": total,
            "Positive_26W": positive_count,
            "Negative_26W": negative_count,
            "Positive_Rate_26W": positive_rate,
            "Average_Return_26W": average_return_26,
            "Median_Return_26W": median_return_26,
            "Average_Return_52W": average_return_52,
            "Median_Return_52W": median_return_52,
            "Fallback_Count": fallback_count
        }

    def analyze_method(
        self,
        data,
        method
    ):

        # IMPORTANTE:
        # Los retornos futuros deben calcularse sobre TODAS
        # las semanas antes de seleccionar el representante.
        prepared_data = (
            self.calculate_forward_returns(
                data.copy()
            )
        )

        representatives = (
            self.select_representatives(
                prepared_data,
                method
            )
        )

        if representatives.empty:
            return representatives

        representatives = (
            self.classify_results(
                representatives
            )
        )

        return representatives

    def compare_methods(self, data):

        methods = [
            "MIN_VFI",
            "FIRST_RECOVERY",
            "BEST_RECOVERY"
        ]

        results = {}

        for method in methods:

            method_data = self.analyze_method(
                data,
                method
            )

            results[method] = method_data

        return results

    def methods_summary(
        self,
        method_results
    ):

        if not method_results:
            return pd.DataFrame()

        rows = []

        for method, method_data in method_results.items():

            summary = self.summarize_method(
                method_data,
                method
            )

            rows.append(
                summary
            )

        return pd.DataFrame(
            rows
        )

    def horizon_summary(
        self,
        method_results
    ):

        rows = []

        for method, method_data in method_results.items():

            if method_data is None:
                continue

            if method_data.empty:
                continue

            for horizon in self.horizons:

                column = (
                    f"Return_{horizon}W"
                )

                if column not in method_data.columns:
                    continue

                returns = (
                    method_data[column]
                    .dropna()
                )

                if returns.empty:
                    continue

                positive_count = (
                    returns
                    > self.positive_return_threshold
                ).sum()

                positive_rate = (
                    positive_count
                    / len(returns)
                    * 100.0
                )

                rows.append(
                    {
                        "Method": method,
                        "Horizon": f"{horizon}W",
                        "Count": len(returns),
                        "Average_Return": returns.mean(),
                        "Median_Return": returns.median(),
                        "Positive_Count": positive_count,
                        "Positive_Rate": positive_rate,
                        "Best_Return": returns.max(),
                        "Worst_Return": returns.min()
                    }
                )

        return pd.DataFrame(
            rows
        )

    def analyze(self, data):

        self.validate_data(
            data
        )

        method_results = (
            self.compare_methods(
                data
            )
        )

        summary = (
            self.methods_summary(
                method_results
            )
        )

        horizon_summary = (
            self.horizon_summary(
                method_results
            )
        )

        return {
            "methods": method_results,
            "summary": summary,
            "horizon_summary": horizon_summary
        }