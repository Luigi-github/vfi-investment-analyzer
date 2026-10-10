import numpy as np
import pandas as pd

from scripts.run_crypto_accumulation_breakout_screen import analyze_symbol


def candles(periods: int, freq: str, start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC", name="Date")
    x = np.arange(periods, dtype=float)
    close = 100 + 0.03 * x + 4 * np.sin(x / 7)
    open_ = close - 0.25 * np.cos(x / 3)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + 0.8,
            "Low": np.minimum(open_, close) - 0.8,
            "Close": close,
            "Volume": 10000 + 1500 * (1 + np.sin(x / 5)),
        },
        index=idx,
    )


def test_short_weekly_history_does_not_erase_daily_vfi_analysis():
    daily = candles(300, "D")
    weekly = candles(60, "7D")
    result = analyze_symbol("TESTUSDT", daily, weekly)

    assert result["Analysis_Status"] == "PARTIAL_ANALYSIS"
    assert result["VFI_Analysis_Status"] == "ANALYZED"
    assert result["Divergence_Analysis_Status"] == "INSUFFICIENT_HISTORY"
    assert result["Daily_Bars"] == 300
    assert result["Weekly_Bars"] == 60
    assert pd.notna(result["VFI_1D_Value"])
    assert result["Divergence_1W_Setup_Type"] == "INSUFFICIENT_HISTORY"
    assert not bool(result["Combined_Setup"])


def test_short_daily_history_does_not_erase_weekly_divergence_analysis():
    daily = candles(100, "D")
    weekly = candles(130, "7D")
    result = analyze_symbol("TESTUSDT", daily, weekly)

    assert result["Analysis_Status"] == "PARTIAL_ANALYSIS"
    assert result["VFI_Analysis_Status"] == "INSUFFICIENT_HISTORY"
    assert result["Divergence_Analysis_Status"] == "ANALYZED"
    assert result["VFI_1D_Setup_Type"] == "INSUFFICIENT_HISTORY"
    assert result["Divergence_1W_Setup_Type"] != "INSUFFICIENT_HISTORY"
    assert not bool(result["Combined_Setup"])
