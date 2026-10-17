"""Regression tests for weekly EMA optimization helpers."""
import numpy as np
import pandas as pd
import pytest

from scripts.optimize_ema_periods import candidate_periods, simulate_strategy, training_score


def test_candidate_grid_contains_5_20_50_and_keeps_ordering():
    periods = candidate_periods()
    assert (5, 20, 50) in periods
    assert periods
    assert all(0 < short < medium < long for short, medium, long in periods)


def test_simulated_position_uses_next_bar_after_signal():
    index = pd.date_range("2010-01-01", periods=120, freq="W-FRI")
    close = pd.Series(np.linspace(50.0, 100.0, len(index)), index=index)
    vfi = pd.Series(np.linspace(-10.0, 0.0, len(index)), index=index)
    data = pd.DataFrame({"Close": close, "VFI": vfi}, index=index)

    result = simulate_strategy(data, short=3, medium=10, long=20, near_zero=5.0)
    signal_dates = result.index[result["Signal"].eq(1)]
    assert len(signal_dates) > 0
    first_signal = signal_dates[0]
    assert result.loc[first_signal, "Position"] == 0.0
    next_date = result.index[result.index.get_loc(first_signal) + 1]
    assert result.loc[next_date, "Position"] == 1.0


def test_training_score_rejects_too_few_entries():
    metrics = {
        "Bars": 200,
        "Entries": 2,
        "CAGR_Pct": 15.0,
        "Max_Drawdown_Pct": -10.0,
    }
    assert training_score(metrics) == -np.inf


def test_invalid_ema_order_is_rejected():
    index = pd.date_range("2010-01-01", periods=80, freq="W-FRI")
    data = pd.DataFrame({
        "Close": np.linspace(10, 20, len(index)),
        "VFI": np.linspace(-2, 2, len(index)),
    }, index=index)
    with pytest.raises(ValueError, match="short < medium < long"):
        simulate_strategy(data, short=20, medium=5, long=50)
