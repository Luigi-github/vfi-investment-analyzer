"""Rank early accumulation candidates using daily VFI and weekly Volume Divergence.

The intended setup is a rising oscillator that is still below zero (or has
crossed zero only very recently). Intraday timeframes are intentionally absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_zero_approach(
    values: pd.Series,
    prefix: str,
    *,
    lookback: int = 52,
    min_history: int = 20,
    near_zero_fraction: float = 0.15,
    max_abs_near_zero: float | None = None,
    recent_cross_fraction: float = 0.10,
    min_strength_4: float = 0.10,
    min_strength_8: float = 0.25,
) -> dict:
    """Describe a pre-zero-break setup with fixed, auditable rules.

    Strength is the 4- and 8-bar net rise divided by the rolling standard
    deviation of the oscillator. This normalizes VFI and the divergence line,
    whose raw scales are not comparable.
    """
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    series = pd.to_numeric(values, errors="coerce").dropna()
    if len(series) < max(min_history, 9):
        return {
            f"{prefix}_Value": float("nan"),
            f"{prefix}_Slope_1B": float("nan"),
            f"{prefix}_Slope_4B": float("nan"),
            f"{prefix}_Slope_8B": float("nan"),
            f"{prefix}_Range_52B": float("nan"),
            f"{prefix}_Near_Zero": False,
            f"{prefix}_Strong_Positive_Slope": False,
            f"{prefix}_Setup_Type": "INSUFFICIENT_HISTORY",
            f"{prefix}_Setup_Score": 0.0,
            f"{prefix}_Normalized_Slope_4B": float("nan"),
            f"{prefix}_Normalized_Slope_8B": float("nan"),
        }

    value = float(series.iloc[-1])
    prior = float(series.iloc[-2])
    slope_1 = value - float(series.iloc[-2])
    slope_4 = (value - float(series.iloc[-5])) / 4.0
    slope_8 = (value - float(series.iloc[-9])) / 8.0

    roll_min = series.rolling(lookback, min_periods=min_history).min().iloc[-1]
    roll_max = series.rolling(lookback, min_periods=min_history).max().iloc[-1]
    roll_std = series.rolling(lookback, min_periods=min_history).std(ddof=0).iloc[-1]
    if not np.isfinite(roll_min) or not np.isfinite(roll_max):
        roll_min, roll_max = float(series.min()), float(series.max())
    value_range = float(roll_max - roll_min)
    scale = float(roll_std) if np.isfinite(roll_std) and roll_std > 0 else 0.0

    norm_4 = ((value - float(series.iloc[-5])) / scale) if scale else 0.0
    norm_8 = ((value - float(series.iloc[-9])) / scale) if scale else 0.0

    near_limit = value_range * near_zero_fraction
    if max_abs_near_zero is not None:
        near_limit = min(near_limit, max_abs_near_zero)
    near_zero = value_range > 0 and abs(value) <= near_limit
    positive_slopes = slope_1 > 0 and slope_4 > 0 and slope_8 > 0
    strong_slope = (
        positive_slopes
        and norm_4 >= min_strength_4
        and norm_8 >= min_strength_8
    )
    crossed_recently = (
        prior < 0 <= value
        and value_range > 0
        and value <= value_range * recent_cross_fraction
        and slope_1 > 0
        and slope_4 > 0
        and slope_8 > 0
    )

    if value < 0 and near_zero and strong_slope:
        setup_type = "PRE_ZERO_BREAK"
    elif crossed_recently:
        setup_type = "ZERO_CROSS_RECENT"
    else:
        setup_type = "NONE"

    # Continuous 0-100 score: proximity (40), slope strength (40), consistency (20).
    proximity = (
        max(0.0, 1.0 - abs(value) / near_limit) * 40.0
        if near_limit > 0
        else 0.0
    )
    strength = 20.0 * min(max(norm_4, 0.0) / max(min_strength_4, 1e-9), 1.0)
    strength += 20.0 * min(max(norm_8, 0.0) / max(min_strength_8, 1e-9), 1.0)
    consistency = 20.0 if positive_slopes else 0.0
    score = float(min(100.0, proximity + strength + consistency))
    if setup_type == "NONE":
        # Scores help diagnose near misses, but should not outrank valid setups.
        score = min(score, 59.9)

    return {
        f"{prefix}_Value": value,
        f"{prefix}_Slope_1B": slope_1,
        f"{prefix}_Slope_4B": slope_4,
        f"{prefix}_Slope_8B": slope_8,
        f"{prefix}_Range_52B": value_range,
        f"{prefix}_Near_Zero": bool(near_zero),
        f"{prefix}_Strong_Positive_Slope": bool(strong_slope),
        f"{prefix}_Normalized_Slope_4B": float(norm_4),
        f"{prefix}_Normalized_Slope_8B": float(norm_8),
        f"{prefix}_Setup_Type": setup_type,
        f"{prefix}_Setup_Score": score,
    }


def rank_accumulation_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Put dual-indicator setups first, then VFI-only setups, then near misses."""
    if results.empty:
        return results.copy()
    result = results.copy()
    vfi_setup = result.get("VFI_Setup_Type", pd.Series("NONE", index=result.index)).isin(
        ["PRE_ZERO_BREAK", "ZERO_CROSS_RECENT"]
    )
    div_setup = result.get("Divergence_Setup_Type", pd.Series("NONE", index=result.index)).isin(
        ["PRE_ZERO_BREAK", "ZERO_CROSS_RECENT"]
    )
    result["VFI_Setup"] = vfi_setup
    result["Divergence_Setup"] = div_setup
    result["Combined_Setup"] = vfi_setup & div_setup
    result["Setup_Tier"] = np.select(
        [result["Combined_Setup"], vfi_setup],
        [1, 2],
        default=3,
    )
    bullish_div = result.get(
        "Bullish_Divergence_Recent", pd.Series(False, index=result.index)
    ).fillna(False).astype(bool)
    vfi_score = pd.to_numeric(
        result["VFI_Setup_Score"] if "VFI_Setup_Score" in result else pd.Series(0.0, index=result.index),
        errors="coerce",
    ).fillna(0)
    divergence_score = pd.to_numeric(
        result["Divergence_Setup_Score"] if "Divergence_Setup_Score" in result else pd.Series(0.0, index=result.index),
        errors="coerce",
    ).fillna(0)
    result["Combined_Opportunity_Score"] = (
        0.65 * vfi_score
        + 0.35 * divergence_score
        + bullish_div.astype(float) * 5.0
    ).clip(upper=100.0)
    return result.sort_values(
        ["Setup_Tier", "Combined_Opportunity_Score", "VFI_Setup_Score"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
