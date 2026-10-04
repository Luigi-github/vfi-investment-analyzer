import pandas as pd

from app.backtest.multi_asset_recovery_analysis import MultiAssetRecoveryAnalyzer


def make_stage_data():
    index = pd.date_range("2024-01-05", periods=12, freq="W-FRI")
    stages = [
        "BEARISH_FLOW",
        "EARLY_RECOVERY",
        "EARLY_RECOVERY",
        "SUSTAINED_RECOVERY",
        "SUSTAINED_RECOVERY",
        "APPROACHING_ZERO",
        "ZERO_CROSS_UP",
        "ACCUMULATION",
        "BULLISH_FLOW",
        "WEAKENING",
        "SUSTAINED_WEAKENING",
        "DISTRIBUTION",
    ]

    return pd.DataFrame(
        {
            "Close": range(100, 112),
            "VFI": [-8, -7.5, -7.0, -6.0, -5.0, -1.0, 0.5, 2.0, 3.0, 2.0, 1.0, 0.0],
            "VFI_Slope": [0, 0.5, 0.5, 1.0, 1.0, 4.0, 1.5, 1.5, 1.0, -1.0, -1.0, -1.0],
            "VFI_Slope_4W": [0, -0.5, -0.25, 0.5, 0.75, 1.5, 1.875, 1.75, 1.5, 0.5, -0.5, -1.0],
            "VFI_Slope_8W": [0, -0.25, -0.2, 0.25, 0.5, 0.9, 1.1, 1.2, 1.0, 0.5, 0.1, -0.1],
            "VFI_Distance_Zero": [8, 7.5, 7, 6, 5, 1, 0.5, 2, 3, 2, 1, 0],
            "VFI_Range_Position": [0.05] * 12,
            "VFI_Cycle_Stage": stages,
            "Recovery_Cluster_ID": list(range(1, 13)),
        },
        index=index,
    )


def test_early_recovery_does_not_force_four_week_positive():
    data = make_stage_data()
    analyzer = MultiAssetRecoveryAnalyzer()

    events = analyzer.select_causal_cycle_stage_events(
        data,
        stages=("EARLY_RECOVERY", "SUSTAINED_RECOVERY"),
    )

    assert list(events["Cycle_Stage_Event"]) == [
        "EARLY_RECOVERY",
        "SUSTAINED_RECOVERY",
    ]

    early = events.iloc[0]
    sustained = events.iloc[1]

    assert early["VFI_Slope_4W"] <= 0
    assert sustained["VFI_Slope_4W"] > 0


def test_cycle_stage_analysis_compares_sustained_recovery():
    data = make_stage_data()
    analyzer = MultiAssetRecoveryAnalyzer()

    result = analyzer.analyze_cycle_stages(
        {"TEST": data},
        stages=("EARLY_RECOVERY", "SUSTAINED_RECOVERY"),
    )

    summary = result["stage_summary"].set_index("Cycle_Stage")

    assert "EARLY_RECOVERY" in summary.index
    assert "SUSTAINED_RECOVERY" in summary.index
    assert summary.loc["SUSTAINED_RECOVERY", "Average_Slope_4W"] > 0
    assert summary.loc["EARLY_RECOVERY", "Average_Slope_4W"] <= 0
