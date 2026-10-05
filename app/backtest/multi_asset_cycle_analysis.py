import pandas as pd

from app.backtest.cycle_analysis import VFICycleBacktestAnalyzer
from app.backtest.statistical_validation import StatisticalValidationAnalyzer
from app.backtest.statistical_robustness import StatisticalRobustnessAnalyzer


class MultiAssetCycleAnalyzer:
    """Run independent cycle analysis across several assets."""

    def __init__(self, horizons=(4, 8, 12, 26, 52), primary_horizon=26):
        if primary_horizon not in tuple(horizons):
            raise ValueError("primary_horizon must be included in horizons.")
        self.primary_horizon = primary_horizon
        self.analyzer = VFICycleBacktestAnalyzer(horizons=horizons)
        self.statistics = StatisticalValidationAnalyzer()
        self.robustness = StatisticalRobustnessAnalyzer()

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
            "quality_summary": self.analyzer.summarize(cycles, complete_only=True),
            "stage_summary": self._stage_summary(events),
            "stage_statistical_summary": self._stage_statistical_summary(events),
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

    def _stage_statistical_summary(self, events):
        if events.empty:
            return pd.DataFrame()

        primary = f"Return_{self.primary_horizon}W"
        valid_all = events[primary].dropna()
        if valid_all.empty:
            return pd.DataFrame()

        condition_returns = {}
        base_rows = []
        for stage, group in events.groupby("Cycle_Stage_Event"):
            returns = group[primary].dropna()
            if returns.empty:
                continue

            # Compare each stage against the complementary set of valid
            # event returns, so the stage is not included in its own benchmark.
            complement_mask = (
                events["Cycle_Stage_Event"].ne(stage)
                & events[primary].notna()
            )
            benchmark = events.loc[complement_mask, primary].dropna()
            if benchmark.empty:
                continue

            condition_returns[stage] = returns
            summary = self.statistics.summarize(returns)
            bootstrap = self.statistics.bootstrap_mean(returns)
            cluster_bootstrap = self.robustness.bootstrap_clustered_mean(
                returns,
                group["Symbol"],
            )
            excess = self.robustness.bootstrap_excess_independent(
                returns, benchmark
            )
            base_rows.append({
                "Cycle_Stage": stage,
                "Count": summary["Count"],
                "Average_Return": summary["Average_Return"],
                "Median_Return": summary["Median_Return"],
                "Std_Return": summary["Std_Return"],
                "Positive_Rate": summary["Positive_Rate"],
                "Worst_Return": summary["Worst_Return"],
                "Best_Return": summary["Best_Return"],
                "Benchmark_Average_Return": float(benchmark.mean()),
                "Excess_Average_Return": excess["Excess_Mean"],
                "Excess_CI_Lower": excess["CI_Lower"],
                "Excess_CI_Upper": excess["CI_Upper"],
                "Bootstrap_CI_Lower": bootstrap["CI_Lower"],
                "Bootstrap_CI_Upper": bootstrap["CI_Upper"],
                "Cluster_Bootstrap_CI_Lower": cluster_bootstrap["CI_Lower"],
                "Cluster_Bootstrap_CI_Upper": cluster_bootstrap["CI_Upper"],
                "Cluster_Count": cluster_bootstrap["Cluster_Count"],
            })

        significance = self.robustness.test_conditions(condition_returns)
        if significance.empty:
            return pd.DataFrame(base_rows)

        significance = significance[[
            "Condition", "Raw_P_Value", "BH_Q_Value", "BH_Significant_5pct"
        ]].rename(columns={"Condition": "Cycle_Stage"})
        result = pd.DataFrame(base_rows).merge(
            significance, on="Cycle_Stage", how="left"
        )
        return result.sort_values(
            "Average_Return", ascending=False
        ).reset_index(drop=True)

    @staticmethod
    def _asset_summary(cycles):
        if cycles.empty:
            return pd.DataFrame()

        bullish = cycles[
            (cycles["Cycle_Type"] == "FULL_VFI_CYCLE")
            & cycles["Complete"]
        ].copy()
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
