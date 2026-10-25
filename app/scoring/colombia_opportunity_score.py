"""Transparent opportunity ranking for early weekly VFI accumulation.

The score is a research priority, not a forecast or a buy recommendation.
It uses only the current row and trailing history available at that row.
"""
import numpy as np
import pandas as pd


def _rolling_r2(series: pd.Series, window: int = 26) -> pd.Series:
    """R-squared of a linear fit to VFI within each trailing window."""
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    x = np.arange(window, dtype=float)
    for end in range(window - 1, len(values)):
        y = values[end - window + 1:end + 1]
        if not np.isfinite(y).all() or np.std(y) == 0:
            continue
        result[end] = float(np.corrcoef(x, y)[0, 1] ** 2)
    return pd.Series(result, index=series.index, name="VFI_R2_26W")


def add_colombia_opportunity_score(data: pd.DataFrame) -> pd.DataFrame:
    """Add EMAs, trend quality and a cumulative 52-week opportunity score.

    Each week below zero with a sustained rising VFI earns 2 points. A rising
    but not fully sustained negative VFI earns 1 point. Sustained declines
    above zero subtract 1 point per week. Current EMA structure, recent
    slopes, R² and a proportional extension penalty adjust that base score.
    """
    df = data.copy()
    if "VFI" not in df:
        raise ValueError("Missing VFI column")

    vfi = pd.to_numeric(df["VFI"], errors="coerce")
    df["VFI_EMA_Short"] = vfi.ewm(span=5, adjust=False, min_periods=5).mean()
    df["VFI_EMA_Medium"] = vfi.ewm(span=20, adjust=False, min_periods=20).mean()
    df["VFI_EMA_Long"] = vfi.ewm(span=50, adjust=False, min_periods=50).mean()
    df["VFI_R2_26W"] = _rolling_r2(vfi, 26)

    slope_1 = pd.to_numeric(df.get("VFI_Slope", vfi.diff()), errors="coerce")
    slope_4 = pd.to_numeric(df.get("VFI_Slope_4B", (vfi - vfi.shift(4)) / 4), errors="coerce")
    slope_8 = pd.to_numeric(df.get("VFI_Slope_8B", (vfi - vfi.shift(8)) / 8), errors="coerce")

    sustained_up = (slope_4 > 0) & (slope_8 > 0)
    rising_negative = (vfi < 0) & (slope_1 > 0)
    sustained_negative_up = (vfi < 0) & sustained_up & (slope_1 > 0)
    sustained_positive_down = (vfi > 0) & (slope_4 < 0) & (slope_8 < 0)

    df["Weekly_Negative_Sustained_Up_Points"] = np.where(sustained_negative_up, 2.0, 0.0)
    df["Weekly_Negative_Early_Recovery_Points"] = np.where(
        rising_negative & ~sustained_up, 1.0, 0.0
    )
    df["Weekly_Positive_Decline_Penalty"] = np.where(sustained_positive_down, 1.0, 0.0)

    # Score the last 52 complete weekly observations to emphasize recent regimes.
    df["Base_52W_Score"] = (
        df["Weekly_Negative_Sustained_Up_Points"]
        + df["Weekly_Negative_Early_Recovery_Points"]
        - df["Weekly_Positive_Decline_Penalty"]
    ).rolling(52, min_periods=1).sum()

    ema_short = df["VFI_EMA_Short"]
    ema_medium = df["VFI_EMA_Medium"]
    ema_long = df["VFI_EMA_Long"]
    ema_bull = (ema_short > ema_medium) & (ema_medium > ema_long)
    df["EMA_Bullish_Alignment"] = ema_bull
    df["EMA_Short_Cross_Up_Medium"] = (
        (ema_short > ema_medium) & (ema_short.shift(1) <= ema_medium.shift(1))
    )
    df["EMA_Medium_Cross_Up_Long"] = (
        (ema_medium > ema_long) & (ema_medium.shift(1) <= ema_long.shift(1))
    )
    df["EMA_Short_Cross_Down_Medium"] = (
        (ema_short < ema_medium) & (ema_short.shift(1) >= ema_medium.shift(1))
    )
    df["EMA_Medium_Cross_Down_Long"] = (
        (ema_medium < ema_long) & (ema_medium.shift(1) >= ema_long.shift(1))
    )

    df["EMA_Structure_Points"] = (
        ema_bull.astype(float) * 8
        + df["EMA_Short_Cross_Up_Medium"].astype(float) * 4
        + df["EMA_Medium_Cross_Up_Long"].astype(float) * 4
        - df["EMA_Short_Cross_Down_Medium"].astype(float) * 4
        - df["EMA_Medium_Cross_Down_Long"].astype(float) * 6
    )
    df["Slope_Points"] = (
        (slope_1 > 0).astype(float) * 2
        + (slope_4 > 0).astype(float) * 3
        + (slope_8 > 0).astype(float) * 4
        + ((slope_1 > 0) & (slope_4 > 0) & (slope_8 > 0)).astype(float) * 3
        - (slope_1 < 0).astype(float) * 2
        - ((slope_4 < 0) & (slope_8 < 0)).astype(float) * 5
    )
    df["VFI_Quality_Points"] = df["VFI_R2_26W"].fillna(0).clip(0, 1) * 10
    df.loc[~((slope_4 > 0) & (slope_8 > 0)), "VFI_Quality_Points"] *= 0.5

    # Extension is relative to this asset's own VFI history, not a shared raw threshold.
    expanding_p80 = vfi.expanding(min_periods=52).quantile(0.80)
    expanding_p95 = vfi.expanding(min_periods=52).quantile(0.95)
    denominator = (expanding_p95 - expanding_p80).replace(0, np.nan)
    extension = ((vfi - expanding_p80) / denominator).clip(0, 1).fillna(0)
    df["VFI_Extension_Factor"] = extension
    df["Positive_Points_Before_Penalty"] = (
        df["Base_52W_Score"].clip(lower=0)
        + df["EMA_Structure_Points"].clip(lower=0)
        + df["Slope_Points"].clip(lower=0)
        + df["VFI_Quality_Points"].clip(lower=0)
    )
    # Penalty scales with positive points accumulated, reaching 50% at extreme VFI.
    df["VFI_Extension_Penalty"] = (
        df["Positive_Points_Before_Penalty"] * extension * 0.50
    )
    all_emas_positive = (ema_short > 0) & (ema_medium > 0) & (ema_long > 0)
    df["All_EMAs_Positive_Penalty"] = (
        df["Positive_Points_Before_Penalty"] * all_emas_positive.astype(float) * 0.20
    )
    df["Zero_Cross_Up_Penalty"] = (
        ((vfi >= 0) & (vfi.shift(1) < 0)).astype(float) * 3
    )
    lost_uptrend = (slope_4.shift(1) > 0) & (slope_8.shift(1) > 0) & (
        (slope_4 <= 0) | (slope_8 <= 0)
    )
    df["Lost_Uptrend_Penalty"] = lost_uptrend.astype(float) * 5

    df["VFI_Opportunity_Score"] = (
        df["Base_52W_Score"]
        + df["EMA_Structure_Points"]
        + df["Slope_Points"]
        + df["VFI_Quality_Points"]
        - df["VFI_Extension_Penalty"]
        - df["All_EMAs_Positive_Penalty"]
        - df["Zero_Cross_Up_Penalty"]
        - df["Lost_Uptrend_Penalty"]
    )
    df["VFI_Opportunity_Score"] = df["VFI_Opportunity_Score"].replace(
        [np.inf, -np.inf], np.nan
    )
    return df
