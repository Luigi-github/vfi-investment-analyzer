"""Unit tests for the scale-aware, causal accumulation ranking helpers."""
import numpy as np
import pandas as pd

from scripts.study_early_vfi_accumulation import (
    build_asset_summary,
    ema_combinations,
    historical_near_zero_threshold,
    historical_percentile,
)


def test_ema_grid_has_only_ordered_combinations():
    combos = ema_combinations()
    assert len(combos) == 64
    assert all(short < medium < long for short, medium, long in combos)


def test_near_zero_threshold_uses_prior_history_not_current_value():
    prior = pd.Series(np.arange(1.0, 101.0))
    a = historical_near_zero_threshold(pd.concat([prior, pd.Series([0.01])]), 200, 50)
    b = historical_near_zero_threshold(pd.concat([prior, pd.Series([1_000_000.0])]), 200, 50)
    assert np.isclose(a, b)
    assert a > 0


def test_historical_percentile_excludes_current_observation():
    prior = pd.Series(np.arange(100.0))
    hist_a = pd.concat([prior, pd.Series([50.0])])
    hist_b = pd.concat([prior, pd.Series([10_000.0])])
    p_a = historical_percentile(50.0, hist_a, 200, 50)
    p_b = historical_percentile(50.0, hist_b, 200, 50)
    assert np.isclose(p_a, p_b)
    assert 0.0 <= p_a <= 1.0


def test_asset_summary_separates_score_from_eligibility():
    rows = []
    for symbol, score, eligible in [("A", 80.0, True), ("B", 95.0, False)]:
        rows.append({
            "Symbol": symbol, "Accumulation_Score": score,
            "Weekly_State_Score": 20.0, "Weekly_EMA_Score": 15.0,
            "Daily_Divergence_Score": 20.0, "Daily_EMA_Score": 10.0,
            "Persistence_Score": 10.0, "Weekly_Filter_Pass": eligible,
            "Daily_Filter_Pass": eligible, "Final_Filter_Pass": eligible,
            "EMA_short": 3, "EMA_medium": 15, "EMA_long": 40,
            "VFI_130": -1.0, "VFI_Slope_1W": 0.2,
            "VFI_Near_Zero_Threshold": 2.0, "Volume_Divergence_MM": -0.2,
            "Volume_Div_Slope_1D": 0.1, "Divergence_Near_Zero_Threshold": 1.0,
            "VFI_Negative_Positive_Slope_Streak_Weeks": 2,
            "Divergence_Negative_Positive_Slope_Streak_Days": 3,
            "Weekly_Date": "2026-10-16", "Daily_Date": "2026-10-16",
        })
    summary = build_asset_summary(pd.DataFrame(rows)).set_index("Symbol")
    assert bool(summary.loc["A", "Eligible"]) is True
    assert bool(summary.loc["B", "Eligible"]) is False
    assert summary.loc["B", "Accumulation_Score"] > summary.loc["A", "Accumulation_Score"]
