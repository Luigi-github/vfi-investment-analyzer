import pandas as pd
import pytest

from app.backtest.cycle_analysis import VFICycleBacktestAnalyzer


def make_data():
    index = pd.date_range("2024-01-05", periods=14, freq="W-FRI")
    vfi = [-3, -2, -1, 0.5, 2, 3, 2, 1, 0, -1, -2, -1, 0.5, 2]
    stages = [
        "EARLY_RECOVERY",
        "SUSTAINED_RECOVERY",
        "SUSTAINED_RECOVERY",
        "ZERO_CROSS_UP",
        "ACCUMULATION",
        "BULLISH_FLOW",
        "WEAKENING",
        "ZERO_CROSS_DOWN",
        "BEARISH_FLOW",
        "EARLY_RECOVERY",
        "SUSTAINED_RECOVERY",
        "APPROACHING_ZERO",
        "ZERO_CROSS_UP",
        "ACCUMULATION",
    ]
    return pd.DataFrame(
        {
            "Close": [100, 102, 105, 110, 115, 120, 118, 112, 105, 103, 106, 109, 114, 118],
            "VFI": vfi,
            "VFI_Slope": [0, 1, 1, 1.5, 1.5, 1, -1, -1, -1, 1, 1, 1, 1.5, 1],
            "VFI_Slope_4W": [0, 0.2, 0.5, 0.8, 1, 1, 0.5, -0.5, -1, -0.5, 0.2, 0.5, 0.8, 1],
            "VFI_Slope_8W": [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.4, -0.2, -0.6, -0.3, 0.1, 0.3, 0.5, 0.6],
            "VFI_Cycle_Stage": stages,
        },
        index=index,
    )


def test_extract_cycles_creates_complete_and_incomplete_cycles():
    analyzer = VFICycleBacktestAnalyzer()
    cycles = analyzer.extract_cycles(make_data())

    assert len(cycles) == 3
    assert cycles.iloc[0]["Cycle_Type"] == "BULLISH_CYCLE"
    assert cycles.iloc[0]["Complete"] is True
    assert cycles.iloc[-1]["Complete"] is False


def test_bullish_cycle_quality_uses_only_cycle_information():
    analyzer = VFICycleBacktestAnalyzer()
    cycles = analyzer.extract_cycles(make_data())

    bullish = cycles[cycles["Cycle_Type"] == "BULLISH_CYCLE"].iloc[0]

    assert bullish["Weeks_SUSTAINED_RECOVERY"] >= 1
    assert bullish["Peak_Return_From_Start_Pct"] > 0
    assert bullish["Cycle_Quality"] in {"EXCEPTIONAL", "STRONG", "MODERATE"}


def test_stage_entry_is_causal_and_only_counts_stage_start():
    analyzer = VFICycleBacktestAnalyzer()
    events = analyzer.stage_entry_analysis(
        make_data(),
        stages=("EARLY_RECOVERY", "SUSTAINED_RECOVERY", "ACCUMULATION"),
    )

    assert list(events["Cycle_Stage_Event"]) == [
        "EARLY_RECOVERY",
        "SUSTAINED_RECOVERY",
        "ACCUMULATION",
        "EARLY_RECOVERY",
        "SUSTAINED_RECOVERY",
        "ACCUMULATION",
    ]
    assert events["Causal_Event"].all()
    assert events.iloc[1]["VFI_Slope_4W"] > 0


def test_validation_rejects_missing_columns():
    analyzer = VFICycleBacktestAnalyzer()
    with pytest.raises(ValueError, match="Missing required columns"):
        analyzer.extract_cycles(pd.DataFrame({"Close": [1, 2]}))
