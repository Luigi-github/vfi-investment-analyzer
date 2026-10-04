import pandas as pd

from app.backtest.cycle_analysis import VFICycleBacktestAnalyzer


class MultiAssetCycleAnalyzer:
    """Run independent cycle analysis across several assets."""

    def __init__(self, horizons=(4, 8, 12, 26, 52), primary_horizon=26):
        if primary_horizon not in tuple(horizons):
            raise ValueError("primary_horizon must be included in horizons.")
        self.primary_horizon = primary_horizon
        self.analyzer = VFICycleBacktestAnalyzer(horizons=horizons)

    def analyze(self, asset_data):
        cycle_frames = []
        event_frames = []

        for symbol, data in asset_data.items():
            cycles = self.analyzer.extract_cycles(data)
            if not cycles.empty:
                cycles = cycles.copy()
                cycles["Symbol"] = symbol
                cycle_frames.append(cycles)

            events = self.analyzer.stage_entry_analysis(data)
            if not events.empty:
                events = events.copy()
                events["Symbol"] = symbol
                event_frames.append(events)

        cycles = (
            pd.concat(cycle_frames, ignore_index=True)
            if cycle_frames else pd.DataFrame()
        )
        events = (
            pd.concat(event_frames, axis=0).sort_index()
            if event_frames else pd.DataFrame()
        )

        return {
            "cycles": cycles,
            "events": events,
            "quality_summary": self.analyzer.summarize(cycles),
            "stage_summary": self._stage_summary(events),
            "asset_summary": self._asset_summary(cycles),
        }

    def _stage_summary(self, events):
        if events.empty:
            return pd.DataFrame()

        primary = f"Return_{self.primary_horizon}W"
        valid = events[events[primary].notna()].copy()
        if valid.empty:
            return pd.DataFrame()

        rows = []
        for stage, group in valid.groupby("Cycle_Stage_Event"):
            returns = group[primary]
            rows.append(
                {
                    "Cycle_Stage": stage,
                    "Events": len(group),
                    "Assets": group["Symbol"].nunique(),
                    f"Average_Return_{self.primary_horizon}W": returns.mean(),
                    f"Median_Return_{self.primary_horizon}W": returns.median(),
                    f"Positive_Rate_{self.primary_horizon}W": returns.gt(0).mean() * 100,
                    "Average_Return_52W": group["Return_52W"].mean()
                    if "Return_52W" in group.columns
                    else None,
                    "Average_VFI": group["VFI"].mean(),
                    "Average_Slope_4W": group["VFI_Slope_4W"].mean(),
                    "Average_Slope_8W": group["VFI_Slope_8W"].mean(),
                }
            )

        return pd.DataFrame(rows).sort_values("Cycle_Stage").reset_index(drop=True)

    @staticmethod
    def _asset_summary(cycles):
        if cycles.empty:
            return pd.DataFrame()

        bullish = cycles[cycles["Cycle_Type"] == "FULL_VFI_CYCLE"].copy()
        if bullish.empty:
            return pd.DataFrame()

        return (
            bullish.groupby("Symbol")
            .agg(
                Full_Cycles=("Cycle_ID", "count"),
                Complete_Cycles=("Complete", "sum"),
                Average_Weeks=("Weeks", "mean"),
                Average_Recovery_Weeks=("Recovery_Weeks", "mean"),
                Average_Cycle_Return_Pct=("Price_Return_Pct", "mean"),
                Median_Cycle_Return_Pct=("Price_Return_Pct", "median"),
                Positive_Cycle_Rate_Pct=(
                    "Price_Return_Pct",
                    lambda values: values.gt(0).mean() * 100,
                ),
            )
            .reset_index()
        )
