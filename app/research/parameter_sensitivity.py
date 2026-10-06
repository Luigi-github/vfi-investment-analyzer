import itertools

import numpy as np
import pandas as pd

from app.analysis.vfi_analyzer import VFIAnalyzer
from app.analysis.vfi_cycle import VFICycleAnalyzer
from app.backtest.multi_asset_recovery_analysis import MultiAssetRecoveryAnalyzer
from app.indicators.vfi import VFIIndicator
from app.scoring.vfi_score import VFIScore


class VFIParameterSensitivityAnalyzer:
    """Walk-forward sensitivity analysis for VFI configuration parameters."""

    PARAMETER_NAMES = ("length", "coef", "vcoef", "signal_length")

    def __init__(self, horizons=(4, 8, 12, 26, 52), primary_horizon=26):
        self.horizons = tuple(horizons)
        self.primary_horizon = primary_horizon

    @classmethod
    def expand_grid(cls, parameter_grid):
        keys = tuple(parameter_grid.keys())
        unknown = set(keys) - set(cls.PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"Unsupported parameters: {sorted(unknown)}")

        values = []
        for key in keys:
            current = parameter_grid[key]
            if not isinstance(current, (list, tuple)):
                current = [current]
            if not current:
                raise ValueError(f"Parameter grid for {key} is empty.")
            values.append(current)

        return [
            dict(zip(keys, combination))
            for combination in itertools.product(*values)
        ]

    def evaluate_configuration(self, weekly_data, configuration):
        indicator = VFIIndicator(**configuration)
        analyzer = VFIAnalyzer()
        scorer = VFIScore()
        cycle_stage = VFICycleAnalyzer()
        recovery = MultiAssetRecoveryAnalyzer(
            horizons=self.horizons,
            primary_horizon=self.primary_horizon,
        )

        features = indicator.calculate(weekly_data)
        features = analyzer.analyze(features)
        features = scorer.calculate(features)
        features = cycle_stage.analyze(features)
        events = recovery.analyze_asset_causal("ASSET", features)

        if events.empty:
            return {
                **configuration,
                "Events": 0,
                "Average_Return": np.nan,
                "Median_Return": np.nan,
                "Positive_Rate": np.nan,
            }

        column = f"Return_{self.primary_horizon}W"
        valid = events[column].dropna()
        return {
            **configuration,
            "Events": int(valid.size),
            "Average_Return": float(valid.mean()) if not valid.empty else np.nan,
            "Median_Return": float(valid.median()) if not valid.empty else np.nan,
            "Positive_Rate": float((valid > 0).mean() * 100.0) if not valid.empty else np.nan,
        }

    def evaluate_grid(self, weekly_data, parameter_grid):
        configurations = self.expand_grid(parameter_grid)
        rows = [
            self.evaluate_configuration(weekly_data, configuration)
            for configuration in configurations
        ]
        return pd.DataFrame(rows)

    def walk_forward(
        self,
        weekly_data,
        parameter_grid,
        initial_train_weeks,
        test_weeks,
        step_weeks,
        minimum_events=10,
    ):
        if initial_train_weeks <= 0 or test_weeks <= 0 or step_weeks <= 0:
            raise ValueError("Walk-forward window sizes must be positive.")

        configurations = self.expand_grid(parameter_grid)
        evaluated = {
            index: self._events_for_configuration(weekly_data, configuration)
            for index, configuration in enumerate(configurations)
        }

        dates = pd.DatetimeIndex(weekly_data.index).sort_values()
        if len(dates) < initial_train_weeks + test_weeks:
            return pd.DataFrame()

        rows = []
        cutoff_position = initial_train_weeks - 1
        window = 1

        while cutoff_position + test_weeks < len(dates):
            cutoff = dates[cutoff_position]
            test_end = dates[cutoff_position + test_weeks]

            candidates = []
            for key, events in evaluated.items():
                train = events.loc[events.index <= cutoff]
                valid = train[f"Return_{self.primary_horizon}W"].dropna()
                if len(valid) < minimum_events:
                    continue
                candidates.append(
                    {
                        "key": key,
                        "train_average": float(valid.mean()),
                        "train_median": float(valid.median()),
                        "train_events": int(len(valid)),
                    }
                )

            if candidates:
                selected = sorted(
                    candidates,
                    key=lambda row: (
                        row["train_average"],
                        row["train_median"],
                        row["train_events"],
                    ),
                    reverse=True,
                )[0]

                selected_events = evaluated[selected["key"]]
                test = selected_events.loc[
                    (selected_events.index > cutoff)
                    & (selected_events.index <= test_end)
                ]
                test_valid = test[f"Return_{self.primary_horizon}W"].dropna()

                configuration = configurations[selected["key"]]
                rows.append(
                    {
                        "Window": window,
                        "Cutoff": cutoff,
                        "Test_End": test_end,
                        **configuration,
                        "Train_Events": selected["train_events"],
                        "Train_Average_Return": selected["train_average"],
                        "Test_Events": int(len(test_valid)),
                        "Test_Average_Return": (
                            float(test_valid.mean())
                            if not test_valid.empty
                            else np.nan
                        ),
                        "Test_Median_Return": (
                            float(test_valid.median())
                            if not test_valid.empty
                            else np.nan
                        ),
                        "Test_Positive_Rate": (
                            float((test_valid > 0).mean() * 100.0)
                            if not test_valid.empty
                            else np.nan
                        ),
                    }
                )

            cutoff_position += step_weeks
            window += 1

        return pd.DataFrame(rows)

    def _events_for_configuration(self, weekly_data, configuration):
        indicator = VFIIndicator(**configuration)
        analyzer = VFIAnalyzer()
        scorer = VFIScore()
        cycle_stage = VFICycleAnalyzer()
        recovery = MultiAssetRecoveryAnalyzer(
            horizons=self.horizons,
            primary_horizon=self.primary_horizon,
        )

        features = indicator.calculate(weekly_data)
        features = analyzer.analyze(features)
        features = scorer.calculate(features)
        features = cycle_stage.analyze(features)
        return recovery.analyze_asset_causal("ASSET", features)
