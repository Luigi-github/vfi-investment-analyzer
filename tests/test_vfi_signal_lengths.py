"""Tests for the two VFI signal lines (EMA spans 20 and 50)."""
import numpy as np
import pandas as pd

from app.indicators.vfi import VFIIndicator


def sample_ohlcv(rows: int = 420) -> pd.DataFrame:
    idx = pd.date_range("2010-01-01", periods=rows, freq="W-FRI")
    close = 100 + np.linspace(0, 25, rows) + 3 * np.sin(np.arange(rows) / 7)
    open_ = close * (1 + 0.002 * np.sin(np.arange(rows) / 3))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = 1_000_000 + 150_000 * (1 + np.sin(np.arange(rows) / 5))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_vfi_has_signal_lengths_20_and_50_with_legacy_alias():
    result = VFIIndicator().calculate(sample_ohlcv())

    expected20 = result["VFI"].ewm(span=20, adjust=False).mean()
    expected50 = result["VFI"].ewm(span=50, adjust=False).mean()

    assert "VFI_Signal_20" in result.columns
    assert "VFI_Signal_50" in result.columns
    assert "VFI_Histogram_20" in result.columns
    assert "VFI_Histogram_50" in result.columns
    pd.testing.assert_series_equal(result["VFI_Signal_20"], expected20, check_names=False)
    pd.testing.assert_series_equal(result["VFI_Signal_50"], expected50, check_names=False)
    pd.testing.assert_series_equal(result["VFI_Signal"], result["VFI_Signal_20"], check_names=False)
    pd.testing.assert_series_equal(
        result["VFI_Histogram_20"], result["VFI"] - result["VFI_Signal_20"], check_names=False
    )
    pd.testing.assert_series_equal(
        result["VFI_Histogram_50"], result["VFI"] - result["VFI_Signal_50"], check_names=False
    )


def test_vfi_signal_length_can_be_configured_and_50_defaults_to_50():
    result = VFIIndicator(signal_length=20).calculate(sample_ohlcv())
    assert np.isclose(
        result["VFI_Signal_20"].dropna().iloc[-1],
        result["VFI"].ewm(span=20, adjust=False).mean().dropna().iloc[-1],
    )
    assert np.isclose(
        result["VFI_Signal_50"].dropna().iloc[-1],
        result["VFI"].ewm(span=50, adjust=False).mean().dropna().iloc[-1],
    )
