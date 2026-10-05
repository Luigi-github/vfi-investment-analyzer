import numpy as np
import pandas as pd

from app.backtest.forward_risk import ForwardRiskAnalyzer
from app.backtest.recovery_event_clustering import RecoveryEventClusterer


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

    def _ensure_causal_recovery_clusters(self, data):
        """Add causal recovery clusters when the caller has not precomputed them.

        Cluster membership is derived only from the current/previous chronological
        EARLY_RECOVERY observations. Existing cluster IDs are preserved so callers
        that already materialize the causal clustering keep their exact data.
        """
        if "Recovery_Cluster_ID" in data.columns:
            return data.copy()

        clusterer = RecoveryEventClusterer()
        events = clusterer.detect_events(data)
        clustered = clusterer.cluster_events_causal(events)

        result = data.copy()
        if clustered.empty:
            result["Recovery_Cluster_ID"] = pd.Series(
                pd.NA,
                index=result.index,
                dtype="Int64",
            )
            return result

        result["Recovery_Cluster_ID"] = pd.Series(
            clustered["Recovery_Cluster_ID"],
            index=clustered.index,
            dtype="Int64",
        )
        return result

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

    def select_causal_non_overlapping_events(self, data):
        """Keep causal representatives whose forward windows do not overlap.

        Selection uses only chronological event dates and the configured
        primary horizon. Forward returns, risk metrics, and future VFI values
        are never used to decide which event is retained.
        """
        if data.empty:
            return data.copy()

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Events must use a DatetimeIndex.")

        ordered = data.sort_index().copy()
        minimum_gap = pd.Timedelta(weeks=self.primary_horizon)

        selected_rows = []
        last_selected_date = None

        for event_date, row in ordered.iterrows():
            if (
                last_selected_date is None
                or event_date >= last_selected_date + minimum_gap
            ):
                selected = row.copy()
                selected["NonOverlapping_Causal"] = True
                selected_rows.append((event_date, selected))
                last_selected_date = event_date

        if not selected_rows:
            return ordered.iloc[0:0].copy()

        result = pd.DataFrame(
            [row for _, row in selected_rows],
            index=[event_date for event_date, _ in selected_rows],
        )
        result.index.name = ordered.index.name
        return result


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
        data = self._ensure_causal_recovery_clusters(data)
        self.validate_data(data)
        df = self.calculate_forward_returns(data.copy())
        df = ForwardRiskAnalyzer(
            horizons=self.horizons
        ).calculate(df)
        representatives = self.select_causal_cluster_representatives(df)
        representatives = self.select_causal_non_overlapping_events(
            representatives
        )

        if representatives.empty:
            return representatives

        representatives = self.classify_results(representatives)
        representatives["Symbol"] = symbol
        return representatives

    def analyze_assets_walk_forward(
        self,
        asset_events,
        initial_train_weeks,
        test_weeks,
        step_weeks
    ):
        """Evaluate causal recovery events in rolling chronological windows."""
        from app.backtest.temporal_validation import TemporalHoldoutAnalyzer

        analyzer = TemporalHoldoutAnalyzer(
            primary_horizon=self.primary_horizon
        )

        rows = []

        for symbol, events in asset_events.items():
            windows = analyzer.walk_forward(
                events,
                initial_train_weeks=initial_train_weeks,
                test_weeks=test_weeks,
                step_weeks=step_weeks,
            )

            for window_number, window in enumerate(windows, start=1):
                row = {
                    "Symbol": symbol,
                    "Window": window_number,
                    "Cutoff": window["cutoff"],
                    "Test_End": window["test_end"],
                }

                for period in ("train", "test"):
                    summary = window[f"{period}_summary"]
                    prefix = period.title()
                    row[f"{prefix}_Events"] = summary["Events"]
                    row[f"{prefix}_Average_Return"] = summary["Average_Return"]
                    row[f"{prefix}_Median_Return"] = summary["Median_Return"]
                    row[f"{prefix}_Positive_Rate"] = summary["Positive_Rate"]

                rows.append(row)

        return pd.DataFrame(rows)

    def summarize_walk_forward_assets(self, walk_forward_results):
        """Summarize out-of-sample consistency by asset."""
        from app.backtest.temporal_validation import TemporalHoldoutAnalyzer

        if walk_forward_results.empty:
            return pd.DataFrame()

        analyzer = TemporalHoldoutAnalyzer(
            primary_horizon=self.primary_horizon
        )
        rows = []

        for symbol, group in walk_forward_results.groupby("Symbol"):
            windows = []
            for _, row in group.sort_values("Window").iterrows():
                windows.append({
                    "test_summary": {
                        "Average_Return": row["Test_Average_Return"]
                    }
                })

            summary = analyzer.summarize_walk_forward(windows)
            summary["Symbol"] = symbol
            rows.append(summary)

        return pd.DataFrame(rows)[[
            "Symbol",
            "Windows",
            "Valid_Windows",
            "Average_Test_Return",
            "Median_Test_Return",
            "Positive_Window_Rate",
            "Worst_Test_Return",
            "Best_Test_Return",
        ]]

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

        df = ForwardRiskAnalyzer(
            horizons=self.horizons
        ).calculate(df)

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

        index_name = combined.index.name
        if index_name is None:
            return combined.sort_values(["Symbol"])

        return combined.sort_values(
            ["Symbol", index_name]
        )

    def asset_summary(
        self,
        combined_data
    ):
        if combined_data.empty:
            return pd.DataFrame()

        primary_column = f"Return_{self.primary_horizon}W"

        if primary_column not in combined_data.columns:
            raise ValueError(
                f"Column '{primary_column}' is required."
            )

        primary_positive_column = f"Positive_{self.primary_horizon}W"
        primary_negative_column = f"Negative_{self.primary_horizon}W"
        primary_rate_column = f"Positive_Rate_{self.primary_horizon}W"
        primary_average_column = f"Average_Return_{self.primary_horizon}W"
        primary_median_column = f"Median_Return_{self.primary_horizon}W"

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
                primary_positive_column: positive_count,
                primary_negative_column: negative_count,
                primary_rate_column: positive_rate,
                primary_average_column:
                    valid[primary_column].mean(),
                primary_median_column:
                    valid[primary_column].median(),
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

        primary_column = f"Return_{self.primary_horizon}W"

        if primary_column not in combined_data.columns:
            raise ValueError(
                f"Column '{primary_column}' is required."
            )

        valid = combined_data[
            combined_data["Primary_Result"].isin(
                ["POSITIVE", "NEGATIVE"]
            )
        ]

        if valid.empty:
            return pd.DataFrame()

        positive_count= (
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
            f"Positive_{self.primary_horizon}W": positive_count,
            f"Negative_{self.primary_horizon}W": negative_count,
            f"Positive_Rate_{self.primary_horizon}W":
                positive_count
                / total
                * 100,
            f"Average_Return_{self.primary_horizon}W":
                valid[primary_column].mean(),
            f"Median_Return_{self.primary_horizon}W":
                valid[primary_column].median(),
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

        primary_column = f"Return_{self.primary_horizon}W"

        if primary_column not in combined_data.columns:
            raise ValueError(
                f"Column '{primary_column}' is required."
            )

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
                    f"Positive_{self.primary_horizon}W": 0,
                    f"Negative_{self.primary_horizon}W": 0,
                    f"Positive_Rate_{self.primary_horizon}W": np.nan,
                    f"Average_Return_{self.primary_horizon}W": np.nan,
                    f"Median_Return_{self.primary_horizon}W": np.nan,
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
                f"Positive_{self.primary_horizon}W": positive_count,
                f"Negative_{self.primary_horizon}W": negative_count,
                f"Positive_Rate_{self.primary_horizon}W":
                    positive_count
                    / total
                    * 100
                    if total > 0
                    else np.nan,
                f"Average_Return_{self.primary_horizon}W":
                    selected[primary_column].mean(),
                f"Median_Return_{self.primary_horizon}W":
                    selected[primary_column].median(),
                "Average_Return_52W":
                    selected[
                        "Return_52W"
                    ].mean()
                    if "Return_52W"
                    in selected.columns
                    else np.nan,
            })

        return pd.DataFrame(rows)

    def select_causal_cycle_stage_events(
        self,
        data,
        stages=(
            "EARLY_RECOVERY",
            "SUSTAINED_RECOVERY",
            "APPROACHING_ZERO",
            "ZERO_CROSS_UP",
        ),
    ):
        """Select the first observed week when each requested cycle stage begins.

        This is intentionally separate from recovery-cluster analysis. Recovery
        clusters are based on EARLY_RECOVERY, while this method lets us compare
        later confirmation stages without changing the definition of an early
        recovery event. The selection is causal: only the current and previous
        stage values are used to detect the stage transition.
        """
        if data.empty:
            return data.copy()

        if "VFI_Cycle_Stage" not in data.columns:
            raise ValueError("Column 'VFI_Cycle_Stage' is required.")

        ordered = data.sort_index().copy()
        previous_stage = ordered["VFI_Cycle_Stage"].shift(1)
        stage_mask = ordered["VFI_Cycle_Stage"].isin(stages)
        entry_mask = stage_mask & (
            previous_stage.isna()
            | previous_stage.ne(ordered["VFI_Cycle_Stage"])
        )

        events = ordered.loc[entry_mask].copy()
        if events.empty:
            return events

        events["Cycle_Stage_Event"] = events["VFI_Cycle_Stage"]
        events["Cycle_Stage_Event_Causal"] = True
        events["Representative_Method"] = "CAUSAL_STAGE_ENTRY"
        return events

    def analyze_cycle_stages(
        self,
        asset_data,
        stages=(
            "EARLY_RECOVERY",
            "SUSTAINED_RECOVERY",
            "APPROACHING_ZERO",
            "ZERO_CROSS_UP",
        ),
    ):
        """Compare causal cycle-stage entries using forward returns.

        Unlike condition_analysis(), this analysis is not restricted to
        recovery-cluster representatives. It therefore allows conditions such
        as VFI_Slope_4W > 0 to be evaluated without creating a logical
        contradiction with EARLY_RECOVERY, whose definition requires
        VFI_Slope_4W <= 0.
        """
        event_frames = []

        for symbol, data in asset_data.items():
            self.validate_data(data)
            df = self.calculate_forward_returns(data.copy())
            df = ForwardRiskAnalyzer(horizons=self.horizons).calculate(df)
            events = self.select_causal_cycle_stage_events(df, stages=stages)
            if events.empty:
                continue
            events["Symbol"] = symbol
            event_frames.append(events)

        if not event_frames:
            return {
                "events": pd.DataFrame(),
                "stage_summary": pd.DataFrame(),
            }

        events = pd.concat(event_frames, axis=0).sort_index()
        primary_column = f"Return_{self.primary_horizon}W"
        valid = events[events[primary_column].notna()].copy()

        rows = []
        for stage, group in valid.groupby("Cycle_Stage_Event"):
            positive = group[primary_column] > self.positive_return_threshold
            rows.append({
                "Cycle_Stage": stage,
                "Events": len(group),
                "Assets": group["Symbol"].nunique(),
                f"Positive_{self.primary_horizon}W": int(positive.sum()),
                f"Negative_{self.primary_horizon}W": int((~positive).sum()),
                f"Positive_Rate_{self.primary_horizon}W": positive.mean() * 100,
                f"Average_Return_{self.primary_horizon}W": group[primary_column].mean(),
                f"Median_Return_{self.primary_horizon}W": group[primary_column].median(),
                "Average_Return_52W": (
                    group["Return_52W"].mean()
                    if "Return_52W" in group.columns
                    else np.nan
                ),
                "Average_VFI": group["VFI"].mean(),
                "Average_Slope_4W": group["VFI_Slope_4W"].mean(),
                "Average_Slope_8W": group["VFI_Slope_8W"].mean(),
            })

        summary = pd.DataFrame(rows)
        if not summary.empty:
            summary = summary.sort_values("Cycle_Stage").reset_index(drop=True)

        return {
            "events": events,
            "stage_summary": summary,
        }

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

    def analyze_condition_discovery_walk_forward(
        self,
        asset_events,
        initial_train_weeks=104,
        test_weeks=52,
        step_weeks=52,
        minimum_events=10,
        top_n=1,
        ranking_metric="Average_Return",
        fee_bps=0.0,
        slippage_bps=0.0,
    ):
        """Run causal condition discovery independently for every asset."""
        from app.backtest.condition_discovery import (
            ConditionDiscoveryAnalyzer,
        )

        results = {}
        summaries = []

        for symbol, events in asset_events.items():
            analyzer = ConditionDiscoveryAnalyzer(
                primary_horizon=self.primary_horizon,
                minimum_events=minimum_events,
                ranking_metric=ranking_metric,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )
            windows = analyzer.walk_forward_discovery(
                events,
                initial_train_weeks=initial_train_weeks,
                test_weeks=test_weeks,
                step_weeks=step_weeks,
                top_n=top_n,
            )
            results[symbol] = windows

            summary = analyzer.summarize_walk_forward(windows)
            if isinstance(summary, dict):
                summary = pd.DataFrame([summary])
            summary.insert(0, "Symbol", symbol)
            summaries.append(summary)

        summary_frame = (
            pd.concat(summaries, ignore_index=True)
            if summaries
            else pd.DataFrame()
        )

        return {
            "asset_windows": results,
            "summary": summary_frame,
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