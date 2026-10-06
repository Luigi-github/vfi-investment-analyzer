import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, roc_auc_score


DEFAULT_FEATURES = [
    "VFI",
    "VFI_Slope",
    "VFI_Slope_4W",
    "VFI_Slope_8W",
    "VFI_Distance_Zero",
    "VFI_Range_Position",
    "VFI_Recovery_From_Min_Pct",
    "VFI_Histogram",
]


class VFIEventMLAnalyzer:
    """Exploratory OOS classifier for positive forward VFI event outcomes."""

    def __init__(
        self,
        primary_horizon=26,
        feature_columns=None,
        random_state=42,
        n_estimators=200,
        max_depth=4,
        minimum_train_events=50,
    ):
        self.primary_horizon = int(primary_horizon)
        self.feature_columns = list(feature_columns or DEFAULT_FEATURES)
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.minimum_train_events = int(minimum_train_events)

    def prepare(self, events):
        return_column = f"Return_{self.primary_horizon}W"
        missing = [
            column for column in self.feature_columns + [return_column]
            if column not in events.columns
        ]
        if missing:
            raise ValueError(f"Missing ML columns: {missing}")

        frame = events[self.feature_columns + [return_column]].copy()
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        frame["Target_Positive"] = (
            frame[return_column] > 0
        ).astype(int)
        return frame

    def fit_predict(self, train, test):
        train = self.prepare(train)
        test = self.prepare(test)

        if len(train) < self.minimum_train_events:
            return None

        if train["Target_Positive"].nunique() < 2:
            return None

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(
            train[self.feature_columns],
            train["Target_Positive"],
        )

        if test.empty:
            return None

        probability = model.predict_proba(
            test[self.feature_columns]
        )[:, 1]
        prediction = (probability >= 0.5).astype(int)

        result = test.copy()
        result["ML_Probability"] = probability
        result["ML_Prediction"] = prediction

        selected_returns = result.loc[
            result["ML_Prediction"].eq(1),
            f"Return_{self.primary_horizon}W",
        ]

        metrics = {
            "Train_Events": int(len(train)),
            "Test_Events": int(len(test)),
            "Predicted_Positive": int(prediction.sum()),
            "Accuracy": float(
                accuracy_score(test["Target_Positive"], prediction)
            ),
            "Balanced_Accuracy": float(
                balanced_accuracy_score(test["Target_Positive"], prediction)
            ),
            "Precision": float(
                precision_score(
                    test["Target_Positive"],
                    prediction,
                    zero_division=0,
                )
            ),
            "ROC_AUC": float(
                roc_auc_score(test["Target_Positive"], probability)
            )
            if test["Target_Positive"].nunique() == 2
            else np.nan,
            "Selected_Average_Return": (
                float(selected_returns.mean())
                if not selected_returns.empty
                else np.nan
            ),
            "Selected_Positive_Rate": (
                float((selected_returns > 0).mean() * 100.0)
                if not selected_returns.empty
                else np.nan
            ),
        }
        return metrics, result

    def walk_forward(
        self,
        events,
        initial_train_weeks=260,
        test_weeks=52,
        step_weeks=52,
    ):
        if not isinstance(events.index, pd.DatetimeIndex):
            raise ValueError("events must use a DatetimeIndex.")

        ordered = events.sort_index()
        dates = pd.DatetimeIndex(ordered.index).unique().sort_values()
        if len(dates) < initial_train_weeks + test_weeks:
            return pd.DataFrame(), pd.DataFrame()

        rows = []
        predictions = []
        cutoff_position = initial_train_weeks - 1
        window = 1

        while cutoff_position + test_weeks < len(dates):
            cutoff = dates[cutoff_position]
            test_end = dates[cutoff_position + test_weeks]

            train = ordered.loc[ordered.index <= cutoff]
            test = ordered.loc[
                (ordered.index > cutoff)
                & (ordered.index <= test_end)
            ]

            fitted = self.fit_predict(train, test)
            if fitted is not None:
                metrics, predicted = fitted
                rows.append(
                    {
                        "Window": window,
                        "Cutoff": cutoff,
                        "Test_End": test_end,
                        **metrics,
                    }
                )
                predicted = predicted.copy()
                predicted["Window"] = window
                predicted["Cutoff"] = cutoff
                predictions.append(predicted)

            cutoff_position += step_weeks
            window += 1

        summary = pd.DataFrame(rows)
        prediction_frame = (
            pd.concat(predictions)
            if predictions
            else pd.DataFrame()
        )
        return summary, prediction_frame

    @staticmethod
    def summarize(walk_forward_summary):
        if walk_forward_summary.empty:
            return pd.DataFrame()

        numeric = walk_forward_summary.select_dtypes(
            include=["number"]
        )
        return pd.DataFrame(
            [{
                "Windows": len(walk_forward_summary),
                "Average_Accuracy": numeric["Accuracy"].mean(),
                "Average_Balanced_Accuracy": numeric["Balanced_Accuracy"].mean(),
                "Average_Precision": numeric["Precision"].mean(),
                "Average_ROC_AUC": numeric["ROC_AUC"].mean(),
                "Average_Selected_Return": numeric[
                    "Selected_Average_Return"
                ].mean(),
                "Positive_Window_Rate": (
                    numeric["Selected_Average_Return"].gt(0).mean() * 100.0
                ),
            }]
        )
