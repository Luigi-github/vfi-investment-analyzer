import numpy as np
import pandas as pd


class VFICycleBacktestAnalyzer:
    """Cycle-level historical analysis built on existing VFI cycle stages.

    This module does not calculate VFI. It only consumes the columns already
    produced by the indicator/scoring pipeline and evaluates complete cycles
    and their subsequent price behavior.
    """

    DEFAULT_HORIZONS = (4, 8, 12, 26, 52)

    def __init__(self, horizons=None):
        self.horizons = tuple(horizons or self.DEFAULT_HORIZONS)

    def validate_data(self, data):
        required = [
            "Close",
            "VFI",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Cycle_Stage",
        ]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Cycle analysis requires a DatetimeIndex.")

    def add_forward_returns(self, data):
        """Add point-in-time forward price returns for stage-entry analysis."""
        df = data.copy()
        for horizon in self.horizons:
            df[f"Return_{horizon}W"] = (
                (df["Close"].shift(-horizon) / df["Close"]) - 1.0
            ) * 100.0
        return df

    def _zero_cross_events(self, data):
        previous = data["VFI"].shift(1)
        up = (previous < 0) & (data["VFI"] >= 0)
        down = (previous > 0) & (data["VFI"] <= 0)
        return up, down

    def extract_cycles(self, data):
        """Extract alternating VFI cycles bounded by zero-cross events.

        A cycle starts at a zero-cross event and ends immediately before the
        next opposite zero-cross event. Incomplete cycles are retained but
        marked as incomplete so they cannot be mistaken for finished cycles.
        """
        self.validate_data(data)
        df = data.sort_index().copy()
        zero_up, zero_down = self._zero_cross_events(df)

        events = []
        for date in df.index[zero_up | zero_down]:
            events.append(
                {
                    "Date": date,
                    "Event": "ZERO_CROSS_UP" if zero_up.loc[date] else "ZERO_CROSS_DOWN",
                }
            )

        if not events:
            return pd.DataFrame()

        rows = []
        for number, start_event in enumerate(events, start=1):
            start_date = start_event["Date"]
            start_type = start_event["Event"]
            end_event = events[number] if number < len(events) else None
            end_date = end_event["Date"] if end_event else df.index[-1]

            segment = df.loc[start_date:end_date]
            if end_event is not None:
                segment = segment.iloc[:-1]

            if segment.empty:
                continue

            start = segment.iloc[0]
            end = segment.iloc[-1]
            stage_counts = segment["VFI_Cycle_Stage"].value_counts()

            row = {
                "Cycle_ID": number,
                "Cycle_Type": (
                    "BULLISH_CYCLE" if start_type == "ZERO_CROSS_UP"
                    else "BEARISH_CYCLE"
                ),
                "Start_Date": segment.index[0],
                "End_Date": segment.index[-1],
                "Boundary_End_Date": end_date,
                "Complete": end_event is not None,
                "Weeks": len(segment),
                "Start_VFI": start["VFI"],
                "End_VFI": end["VFI"],
                "Min_VFI": segment["VFI"].min(),
                "Max_VFI": segment["VFI"].max(),
                "Average_VFI": segment["VFI"].mean(),
                "Max_4W_Slope": segment["VFI_Slope_4W"].max(),
                "Max_8W_Slope": segment["VFI_Slope_8W"].max(),
                "Min_4W_Slope": segment["VFI_Slope_4W"].min(),
                "Min_8W_Slope": segment["VFI_Slope_8W"].min(),
                "Price_Return_Pct": (
                    (end["Close"] / start["Close"]) - 1.0
                ) * 100.0,
                "Peak_Price": segment["Close"].max(),
                "Trough_Price": segment["Close"].min(),
                "Peak_Return_From_Start_Pct": (
                    (segment["Close"].max() / start["Close"]) - 1.0
                ) * 100.0,
                "Worst_Return_From_Start_Pct": (
                    (segment["Close"].min() / start["Close"]) - 1.0
                ) * 100.0,
                "Stage_Count": len(stage_counts),
            }

            for stage in (
                "BEARISH_FLOW",
                "EARLY_RECOVERY",
                "SUSTAINED_RECOVERY",
                "APPROACHING_ZERO",
                "ZERO_CROSS_UP",
                "ACCUMULATION",
                "BULLISH_FLOW",
                "WEAKENING",
                "SUSTAINED_WEAKENING",
                "DISTRIBUTION",
                "ZERO_CROSS_DOWN",
            ):
                row[f"Weeks_{stage}"] = int(stage_counts.get(stage, 0))

            rows.append(row)

        result = pd.DataFrame(rows)
        if not result.empty:
            result["Recovery_Weeks"] = (
                result["Weeks_EARLY_RECOVERY"]
                + result["Weeks_SUSTAINED_RECOVERY"]
                + result["Weeks_APPROACHING_ZERO"]
            )
            result["Accumulation_Weeks"] = (
                result["Weeks_ZERO_CROSS_UP"]
                + result["Weeks_ACCUMULATION"]
                + result["Weeks_BULLISH_FLOW"]
            )
            result["Cycle_Quality"] = result.apply(
                self._classify_cycle_quality, axis=1
            )
        return result

    @staticmethod
    def _classify_cycle_quality(row):
        if row["Cycle_Type"] != "BULLISH_CYCLE":
            return "NOT_BULLISH"

        if not row["Complete"]:
            return "INCOMPLETE"

        recovery = row["Recovery_Weeks"]
        sustained = row["Weeks_SUSTAINED_RECOVERY"]
        peak_return = row["Peak_Return_From_Start_Pct"]
        final_return = row["Price_Return_Pct"]

        if sustained > 0 and peak_return >= 30 and final_return > 0:
            return "EXCEPTIONAL"
        if sustained > 0 and peak_return >= 10 and final_return > 0:
            return "STRONG"
        if recovery > 0 and peak_return > 0 and final_return > 0:
            return "MODERATE"
        if recovery > 0:
            return "WEAK"
        return "FAILED"

    def stage_entry_analysis(self, data, stages=None):
        """Evaluate the first causal week of each requested stage."""
        self.validate_data(data)
        df = self.add_forward_returns(data.sort_index())

        stages = tuple(stages or (
            "EARLY_RECOVERY",
            "SUSTAINED_RECOVERY",
            "APPROACHING_ZERO",
            "ZERO_CROSS_UP",
            "ACCUMULATION",
        ))

        previous = df["VFI_Cycle_Stage"].shift(1)
        entry = (
            df["VFI_Cycle_Stage"].isin(stages)
            & (
                previous.isna()
                | previous.ne(df["VFI_Cycle_Stage"])
            )
        )

        events = df.loc[entry].copy()
        if events.empty:
            return events

        events["Cycle_Stage_Event"] = events["VFI_Cycle_Stage"]
        events["Causal_Event"] = True
        return events

    def summarize(self, cycles):
        """Summarize completed bullish cycles by quality."""
        if cycles.empty:
            return pd.DataFrame()

        bullish = cycles[cycles["Cycle_Type"] == "BULLISH_CYCLE"].copy()
        if bullish.empty:
            return pd.DataFrame()

        rows = []
        for quality, group in bullish.groupby("Cycle_Quality"):
            rows.append(
                {
                    "Cycle_Quality": quality,
                    "Cycles": len(group),
                    "Average_Weeks": group["Weeks"].mean(),
                    "Average_Recovery_Weeks": group["Recovery_Weeks"].mean(),
                    "Average_Peak_Return_Pct": group[
                        "Peak_Return_From_Start_Pct"
                    ].mean(),
                    "Average_Cycle_Return_Pct": group[
                        "Price_Return_Pct"
                    ].mean(),
                    "Median_Cycle_Return_Pct": group[
                        "Price_Return_Pct"
                    ].median(),
                    "Positive_Cycle_Rate_Pct": (
                        (group["Price_Return_Pct"] > 0).mean() * 100
                    ),
                }
            )

        return pd.DataFrame(rows).sort_values(
            "Average_Cycle_Return_Pct", ascending=False
        ).reset_index(drop=True)
