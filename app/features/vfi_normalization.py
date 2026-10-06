import numpy as np
import pandas as pd


def add_vfi_normalized_features(
    data: pd.DataFrame,
    zscore_window: int = 104,
    percentile_window: int = 104,
) -> pd.DataFrame:
    """Add scale-aware VFI features without changing the original VFI."""
    if "VFI" not in data.columns:
        raise ValueError("VFI column is required.")

    if zscore_window <= 1 or percentile_window <= 1:
        raise ValueError("Normalization windows must be greater than 1.")

    df = data.copy()
    rolling = df["VFI"].rolling(zscore_window, min_periods=zscore_window)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    df["VFI_ZScore"] = (df["VFI"] - mean) / std.replace(0, np.nan)

    def percentile_last(values):
        series = pd.Series(values)
        return float(series.rank(pct=True).iloc[-1] * 100.0)

    df["VFI_Percentile"] = (
        df["VFI"]
        .rolling(percentile_window, min_periods=percentile_window)
        .apply(percentile_last, raw=False)
    )

    if "VFI_Rolling_Min" in df.columns and "VFI_Rolling_Max" in df.columns:
        span = df["VFI_Rolling_Max"] - df["VFI_Rolling_Min"]
        df["VFI_Normalized_Range"] = (
            (df["VFI"] - df["VFI_Rolling_Min"]) / span.replace(0, np.nan)
        )
        df["VFI_Distance_From_Historical_Extreme"] = df[
            ["VFI_Distance_From_Min", "VFI_Distance_From_Max"]
        ].min(axis=1) if {"VFI_Distance_From_Min", "VFI_Distance_From_Max"}.issubset(df.columns) else np.nan

    return df
