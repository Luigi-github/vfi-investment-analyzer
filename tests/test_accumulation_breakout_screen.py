import numpy as np
import pandas as pd

from app.research.accumulation_breakout_screen import (
    rank_accumulation_candidates,
    score_zero_approach,
)


def rising_negative_series(end=-0.5, length=60):
    values = -7.0 + np.sin(np.arange(length) / 3.0) * 2.0
    values[-9:] = np.linspace(-5.0, end, 9)
    return pd.Series(values)


def test_pre_zero_break_requires_negative_near_zero_and_strong_positive_slopes():
    result = score_zero_approach(
        rising_negative_series(),
        "VFI_1D",
        lookback=52,
        min_history=20,
        near_zero_fraction=0.15,
        max_abs_near_zero=2.0,
    )
    assert result["VFI_1D_Setup_Type"] == "PRE_ZERO_BREAK"
    assert result["VFI_1D_Value"] < 0
    assert result["VFI_1D_Near_Zero"]
    assert result["VFI_1D_Strong_Positive_Slope"]
    assert result["VFI_1D_Slope_1B"] > 0
    assert result["VFI_1D_Slope_4B"] > 0
    assert result["VFI_1D_Slope_8B"] > 0


def test_high_positive_oscillator_is_not_an_early_breakout_candidate():
    values = pd.Series(np.linspace(-8.0, 6.0, 60))
    result = score_zero_approach(values, "Divergence_1W", lookback=52, min_history=20)
    assert result["Divergence_1W_Setup_Type"] == "NONE"


def test_recent_zero_cross_is_labelled_separately():
    values = rising_negative_series(end=0.1)
    result = score_zero_approach(values, "VFI_1D", lookback=52, min_history=20)
    assert result["VFI_1D_Setup_Type"] == "ZERO_CROSS_RECENT"


def test_insufficient_history_is_not_ranked_as_setup():
    result = score_zero_approach(pd.Series([-3.0, -2.0, -1.0]), "VFI_1D")
    assert result["VFI_1D_Setup_Type"] == "INSUFFICIENT_HISTORY"
    assert result["VFI_1D_Setup_Score"] == 0.0


def test_combined_daily_vfi_weekly_divergence_setup_ranks_first():
    frame = pd.DataFrame(
        [
            {
                "Symbol": "VFI_ONLY",
                "VFI_Setup_Type": "PRE_ZERO_BREAK",
                "Divergence_Setup_Type": "NONE",
                "VFI_Setup_Score": 95.0,
                "Divergence_Setup_Score": 30.0,
                "Bullish_Divergence_Recent": False,
            },
            {
                "Symbol": "BOTH",
                "VFI_Setup_Type": "PRE_ZERO_BREAK",
                "Divergence_Setup_Type": "PRE_ZERO_BREAK",
                "VFI_Setup_Score": 75.0,
                "Divergence_Setup_Score": 70.0,
                "Bullish_Divergence_Recent": True,
            },
            {
                "Symbol": "NONE",
                "VFI_Setup_Type": "NONE",
                "Divergence_Setup_Type": "NONE",
                "VFI_Setup_Score": 50.0,
                "Divergence_Setup_Score": 50.0,
                "Bullish_Divergence_Recent": False,
            },
        ]
    )
    result = rank_accumulation_candidates(frame)
    assert result.iloc[0]["Symbol"] == "BOTH"
    assert bool(result.iloc[0]["Combined_Setup"])
    assert result.iloc[1]["Symbol"] == "VFI_ONLY"
    assert result.iloc[2]["Symbol"] == "NONE"
