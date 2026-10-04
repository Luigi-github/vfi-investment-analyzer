import numpy as np
import pandas as pd


class RecoverySignalAnalyzer:

    def __init__(
        self,
        horizons=(4, 8, 12, 26, 52),
        primary_horizon=26,
        positive_return_threshold=0.0
    ):
        self.horizons = tuple(horizons)
        self.primary_horizon = primary_horizon
        self.positive_return_threshold = positive_return_threshold

        if self.primary_horizon not in self.horizons:
            raise ValueError(
                "primary_horizon must be included in horizons."
            )

    # ==========================================================
    # 1. DETECTAR EVENTOS EARLY_RECOVERY
    # ==========================================================

    def detect_early_recovery_events(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        required_columns = [
            "VFI",
            "VFI_State",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Distance_Zero",
            "VFI_Range_Position",
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

        df = data.copy()

        events = df[
            df["VFI_Cycle_Stage"].eq("EARLY_RECOVERY")
        ].copy()

        return events

    # ==========================================================
    # 2. CALCULAR RETORNOS FUTUROS
    # ==========================================================

    def calculate_forward_returns(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        df = data.copy()

        if "Close" not in df.columns:
            raise ValueError(
                "Column 'Close' is required."
            )

        for horizon in self.horizons:

            future_price = df["Close"].shift(-horizon)

            df[f"Return_{horizon}W"] = (
                (future_price / df["Close"]) - 1.0
            ) * 100.0

        return df

    # ==========================================================
    # 3. CREAR DATASET DE EVENTOS
    # ==========================================================

    def build_event_dataset(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        df = self.calculate_forward_returns(data)

        events = df[
            df["VFI_Cycle_Stage"].eq("EARLY_RECOVERY")
        ].copy()

        if events.empty:
            return events

        # ------------------------------------------------------
        # Resultado principal
        # ------------------------------------------------------

        primary_column = f"Return_{self.primary_horizon}W"

        if primary_column in events.columns:
            events["Primary_Return"] = events[
                primary_column
            ]
        else:
            raise ValueError(
                f"Column '{primary_column}' is required."
            )

        # ------------------------------------------------------
        # Clasificación según retorno
        # ------------------------------------------------------

        events["Primary_Result"] = np.where(
            events["Primary_Return"].isna(),
            "INSUFFICIENT_DATA",
            np.where(
                events["Primary_Return"]
                > self.positive_return_threshold,
                "POSITIVE",
                "NEGATIVE"
            )
        )

        # ------------------------------------------------------
        # Resultado 52 semanas
        # ------------------------------------------------------

        if "Return_52W" in events.columns:

            events["Long_Term_Result"] = np.where(
                events["Return_52W"].isna(),
                "INSUFFICIENT_DATA",
                np.where(
                    events["Return_52W"]
                    > self.positive_return_threshold,
                    "POSITIVE",
                    "NEGATIVE"
                )
            )

        # ------------------------------------------------------
        # Resultado 12 semanas
        # ------------------------------------------------------

        if "Return_12W" in events.columns:

            events["Medium_Term_Result"] = np.where(
                events["Return_12W"].isna(),
                "INSUFFICIENT_DATA",
                np.where(
                    events["Return_12W"]
                    > self.positive_return_threshold,
                    "POSITIVE",
                    "NEGATIVE"
                )
            )

        return events

    # ==========================================================
    # 4. RESUMEN DE FEATURES
    # ==========================================================

    def feature_summary(
        self,
        events: pd.DataFrame
    ) -> pd.DataFrame:

        features = [
            "VFI",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Distance_Zero",
            "VFI_Range_Position",
        ]

        available_features = [
            feature
            for feature in features
            if feature in events.columns
        ]

        valid_events = events[
            events["Primary_Result"].isin(
                ["POSITIVE", "NEGATIVE"]
            )
        ].copy()

        rows = []

        for feature in available_features:

            positive = valid_events.loc[
                valid_events["Primary_Result"] == "POSITIVE",
                feature
            ].dropna()

            negative = valid_events.loc[
                valid_events["Primary_Result"] == "NEGATIVE",
                feature
            ].dropna()

            row = {
                "Feature": feature,

                "Positive_Count": len(positive),
                "Negative_Count": len(negative),

                "Positive_Mean": positive.mean(),
                "Negative_Mean": negative.mean(),

                "Positive_Median": positive.median(),
                "Negative_Median": negative.median(),

                "Positive_Min": positive.min(),
                "Positive_Max": positive.max(),

                "Negative_Min": negative.min(),
                "Negative_Max": negative.max(),

                "Mean_Difference":
                    positive.mean() - negative.mean()
            }

            rows.append(row)

        return pd.DataFrame(rows)

    # ==========================================================
    # 5. ANALIZAR UNA CONDICIÓN
    # ==========================================================

    def analyze_condition(
        self,
        events: pd.DataFrame,
        condition_name: str,
        condition
    ) -> dict:

        valid_events = events[
            events["Primary_Result"].isin(
                ["POSITIVE", "NEGATIVE"]
            )
        ].copy()

        selected = valid_events[
            condition(valid_events)
        ]

        if selected.empty:
            return {
                "Condition": condition_name,
                "Count": 0,
                "Positive_Count": 0,
                "Positive_Rate": np.nan,
                f"Average_Return_{self.primary_horizon}W": np.nan,
                f"Median_Return_{self.primary_horizon}W": np.nan,
            }

        primary_column = f"Return_{self.primary_horizon}W"
        returns = selected[primary_column].dropna()

        positive_count = (
            selected["Primary_Result"]
            .eq("POSITIVE")
            .sum()
        )

        positive_rate = (
            positive_count / len(selected)
        ) * 100

        return {
            "Condition": condition_name,
            "Count": len(selected),
            "Positive_Count": positive_count,
            "Positive_Rate": positive_rate,
            f"Average_Return_{self.primary_horizon}W": returns.mean(),
            f"Median_Return_{self.primary_horizon}W": returns.median(),
        }

    # ==========================================================
    # 6. ANALIZAR CONDICIONES INDIVIDUALES
    # ==========================================================

    def analyze_individual_conditions(
        self,
        events: pd.DataFrame
    ) -> pd.DataFrame:

        conditions = {

            "VFI < 0": (
                lambda df:
                df["VFI"] < 0
            ),

            "VFI < -5": (
                lambda df:
                df["VFI"] < -5
            ),

            "VFI near zero": (
                lambda df:
                df["VFI_Distance_Zero"] <= 1
            ),

            "Slope > 0": (
                lambda df:
                df["VFI_Slope"] > 0
            ),

            "Slope > 0.5": (
                lambda df:
                df["VFI_Slope"] > 0.5
            ),

            "Slope 4W > 0": (
                lambda df:
                df["VFI_Slope_4W"] > 0
            ),

            "Slope 8W > 0": (
                lambda df:
                df["VFI_Slope_8W"] > 0
            ),

            "Range position <= 20%": (
                lambda df:
                df["VFI_Range_Position"] <= 0.20
            ),

            "Range position <= 30%": (
                lambda df:
                df["VFI_Range_Position"] <= 0.30
            ),

            "Slope > 0 AND Slope 8W > 0": (
                lambda df:
                (df["VFI_Slope"] > 0)
                &
                (df["VFI_Slope_8W"] > 0)
            ),

            "Slope 4W > 0 AND Slope 8W > 0": (
                lambda df:
                (df["VFI_Slope_4W"] > 0)
                &
                (df["VFI_Slope_8W"] > 0)
            ),

            "Slope > 0 AND Range <= 20%": (
                lambda df:
                (df["VFI_Slope"] > 0)
                &
                (df["VFI_Range_Position"] <= 0.20)
            ),

            "Slope 8W > 0 AND Range <= 20%": (
                lambda df:
                (df["VFI_Slope_8W"] > 0)
                &
                (df["VFI_Range_Position"] <= 0.20)
            ),
        }

        rows = []

        for name, condition in conditions.items():

            result = self.analyze_condition(
                events,
                name,
                condition
            )

            rows.append(result)

        return pd.DataFrame(rows)

    # ==========================================================
    # 7. COMBINACIONES DE FEATURES
    # ==========================================================

    def analyze_feature_combinations(
        self,
        events: pd.DataFrame
    ) -> pd.DataFrame:

        valid_events = events[
            events["Primary_Result"].isin(
                ["POSITIVE", "NEGATIVE"]
            )
        ].copy()

        conditions = {

            "Strong short + medium recovery":
                (
                    (valid_events["VFI_Slope"] > 0)
                    &
                    (valid_events["VFI_Slope_4W"] > 0)
                    &
                    (valid_events["VFI_Slope_8W"] > 0)
                ),

            "Positive slope + positive 8W":
                (
                    (valid_events["VFI_Slope"] > 0)
                    &
                    (valid_events["VFI_Slope_8W"] > 0)
                ),

            "Positive slope + positive 4W":
                (
                    (valid_events["VFI_Slope"] > 0)
                    &
                    (valid_events["VFI_Slope_4W"] > 0)
                ),

            "Deep VFI + positive slope":
                (
                    (valid_events["VFI"] < -5)
                    &
                    (valid_events["VFI_Slope"] > 0)
                ),

            "Deep VFI + positive 8W":
                (
                    (valid_events["VFI"] < -5)
                    &
                    (valid_events["VFI_Slope_8W"] > 0)
                ),

            "Near lower range + positive slope":
                (
                    (valid_events["VFI_Range_Position"] <= 0.20)
                    &
                    (valid_events["VFI_Slope"] > 0)
                ),

            "Near lower range + positive 8W":
                (
                    (valid_events["VFI_Range_Position"] <= 0.20)
                    &
                    (valid_events["VFI_Slope_8W"] > 0)
                ),

            "Deep VFI + positive slope + positive 8W":
                (
                    (valid_events["VFI"] < -5)
                    &
                    (valid_events["VFI_Slope"] > 0)
                    &
                    (valid_events["VFI_Slope_8W"] > 0)
                ),

            "Lower range + positive slope + positive 8W":
                (
                    (valid_events["VFI_Range_Position"] <= 0.20)
                    &
                    (valid_events["VFI_Slope"] > 0)
                    &
                    (valid_events["VFI_Slope_8W"] > 0)
                ),
        }

        rows = []

        for name, mask in conditions.items():

            selected = valid_events[mask]

            if selected.empty:
                rows.append({
                    "Condition": name,
                    "Count": 0,
                    "Positive_Count": 0,
                    "Positive_Rate": np.nan,
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

            primary_returns = (
                selected[f"Return_{self.primary_horizon}W"]
                .dropna()
            )

            returns_52 = (
                selected["Return_52W"]
                .dropna()
            )

            rows.append({
                "Condition": name,
                "Count": len(selected),
                "Positive_Count": positive_count,
                "Positive_Rate":
                    positive_count / len(selected) * 100,
                f"Average_Return_{self.primary_horizon}W":
                    primary_returns.mean(),
                f"Median_Return_{self.primary_horizon}W":
                    primary_returns.median(),
                "Average_Return_52W":
                    returns_52.mean(),
            })

        return pd.DataFrame(rows)

    # ==========================================================
    # 8. RESUMEN POR HORIZONTE
    # ==========================================================

    def horizon_summary(
        self,
        events: pd.DataFrame
    ) -> pd.DataFrame:

        rows = []

        for horizon in self.horizons:

            column = f"Return_{horizon}W"

            if column not in events.columns:
                continue

            returns = events[column].dropna()

            if returns.empty:
                continue

            rows.append({
                "Horizon": f"{horizon}W",
                "Count": len(returns),
                "Average_Return": returns.mean(),
                "Median_Return": returns.median(),
                "Positive_Count":
                    (returns > 0).sum(),
                "Positive_Rate":
                    (returns > 0).mean() * 100,
                "Best_Return": returns.max(),
                "Worst_Return": returns.min(),
            })

        return pd.DataFrame(rows)

    # ==========================================================
    # 9. ANALISIS COMPLETO
    # ==========================================================

    def analyze(
        self,
        data: pd.DataFrame
    ) -> dict:

        events = self.build_event_dataset(data)

        if events.empty:
            return {
                "events": events,
                "feature_summary": pd.DataFrame(),
                "individual_conditions": pd.DataFrame(),
                "feature_combinations": pd.DataFrame(),
                "horizon_summary": pd.DataFrame(),
            }

        return {
            "events": events,

            "feature_summary":
                self.feature_summary(events),

            "individual_conditions":
                self.analyze_individual_conditions(events),

            "feature_combinations":
                self.analyze_feature_combinations(events),

            "horizon_summary":
                self.horizon_summary(events),
        }