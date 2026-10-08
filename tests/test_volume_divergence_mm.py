import numpy as np
import pandas as pd

from app.indicators.volume_divergence_mm import VolumeDivergenceMM


def test_pine_wma_applies_candle_direction_and_weights():
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    data = pd.DataFrame(
        {
            "Open": [10.0, 12.0, 11.0],
            "High": [11.0, 13.0, 12.0],
            "Low": [9.0, 11.0, 10.0],
            "Close": [11.0, 11.0, 12.0],
            "Volume": [10.0, 20.0, 30.0],
        },
        index=idx,
    )
    indicator = VolumeDivergenceMM()
    result = indicator._pine_wma(
        data["Volume"], data["Open"], data["Close"], 3
    )

    # Pine weights for length 3 are 9, 6, 3.
    expected = (10 * 9 * 1 + 20 * 6 * -1 + 30 * 3 * 1) / 18
    assert np.isclose(result.iloc[-1], expected)


def test_pivot_confirmation_is_not_backdated():
    indicator = VolumeDivergenceMM(lbL=2, lbR=2)
    values = np.array([5.0, 4.0, 1.0, 3.0, 5.0, 6.0])
    pivots = indicator._pivot(values, 2, 2, high=False)

    assert not pivots[2]
    assert not pivots[3]
    assert pivots[4]


def test_indicator_outputs_causal_divergence_columns():
    idx = pd.date_range("2025-01-01", periods=120, freq="D")
    close = 100 + np.sin(np.arange(120) / 5.0) * 5 + np.arange(120) * 0.02
    open_ = close - 0.5
    high = np.maximum(open_, close) + 1
    low = np.minimum(open_, close) - 1
    volume = 1000 + (np.arange(120) % 10) * 25

    data = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )

    result = indicator.calculate(data)

    required = {
        "Volume_Divergence_MM",
        "Volume_Div_Regular_Bull",
        "Volume_Div_Hidden_Bull",
        "Volume_Div_Regular_Bear",
        "Volume_Div_Hidden_Bear",
        "Volume_Div_Bullish",
        "Volume_Div_Bearish",
        "Volume_Div_Signal",
        "Volume_Div_Bullish_Age_Bars",
        "Volume_Div_Bearish_Age_Bars",
    }
    assert required.issubset(result.columns)
