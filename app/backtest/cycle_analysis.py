import pandas as pd


class VFICycleBacktestAnalyzer:
    """Cycle-level historical analysis built on existing VFI cycle stages."""

    DEFAULT_HORIZONS = (4, 8, 12, 26, 52)

    def __init__(self, horizons=None):
        self.horizons = tuple(horizons or self.DEFAULT_HORIZONS)
        if not self.horizons or any(
            not isinstance(horizon, int) or horizon <= 0
            for horizon in self.horizons
        ):
            raise ValueError("Horizons must contain positive integers.")

    def validate_data(self, data):
        required = [
            "Close", "VFI", "VFI_Slope", "VFI_Slope_4W",
            "VFI_Slope_8W", "VFI_Cycle_Stage",
        ]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Cycle analysis requires a DatetimeIndex.")

    def add_forward_returns(self, data):
        """Add causal forward price and total-return observations.

        These are descriptive labels, not executable entry prices. Executable
        trades are handled by the portfolio backtester using the next
        available observation after a signal.
        """
        df = data.copy()
        for horizon in self.horizons:
            df[f"Return_{horizon}W"] = (
                (df["Close"].shift(-horizon) / df["Close"]) - 1.0
            ) * 100.0
            if "Adj Close" in df.columns:
                df[f"Total_Return_{horizon}W"] = (
                    (df["Adj Close"].shift(-horizon) / df["Adj Close"]) - 1.0
                ) * 100.0
        return df

    def _zero_cross_events(self, data):
        previous = data["VFI"].shift(1)
        up = (previous < 0) & (data["VFI"] >= 0)
        down = (previous > 0) & (data["VFI"] <= 0)
        return up, down

    def extract_cycles(self, data):
        self.validate_data(data)
        df = data.sort_index().copy()
        _, zero_down = self._zero_cross_events(df)
        boundary_dates = list(df.index[zero_down])
        if not boundary_dates:
            return pd.DataFrame()

        rows = []
        for number, start_date in enumerate(boundary_dates, start=1):
            end_date = boundary_dates[number] if number < len(boundary_dates) else df.index[-1]
            complete = number < len(boundary_dates)
            segment = df.loc[start_date:end_date]
            if complete:
                segment = segment.iloc[:-1]
            if segment.empty:
                continue

            start, end = segment.iloc[0], segment.iloc[-1]
            stage_counts = segment["VFI_Cycle_Stage"].value_counts()
            row = {
                "Cycle_ID": number,
                "Cycle_Type": "FULL_VFI_CYCLE",
                "Causal_Signal": False,
                "Retrospective_Analysis": True,
                "Start_Date": segment.index[0],
                "End_Date": segment.index[-1],
                "Boundary_End_Date": end_date,
                "Complete": complete,
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
                "Price_Return_Pct": ((end["Close"] / start["Close"]) - 1.0) * 100.0,
                "Peak_Price": segment["Close"].max(),
                "Trough_Price": segment["Close"].min(),
                "Peak_Return_From_Start_Pct": ((segment["Close"].max() / start["Close"]) - 1.0) * 100.0,
                "Worst_Return_From_Start_Pct": ((segment["Close"].min() / start["Close"]) - 1.0) * 100.0,
                "Stage_Count": len(stage_counts),
            }
            if "Adj Close" in segment.columns:
                row["Total_Return_Pct"] = ((end["Adj Close"] / start["Adj Close"]) - 1.0) * 100.0
                row["Peak_Total_Return_From_Start_Pct"] = ((segment["Adj Close"].max() / start["Adj Close"]) - 1.0) * 100.0
                row["Worst_Total_Return_From_Start_Pct"] = ((segment["Adj Close"].min() / start["Adj Close"]) - 1.0) * 100.0

            for stage in (
                "BEARISH_FLOW", "EARLY_RECOVERY", "SUSTAINED_RECOVERY",
                "APPROACHING_ZERO", "ZERO_CROSS_UP", "ACCUMULATION",
                "BULLISH_FLOW", "WEAKENING", "SUSTAINED_WEAKENING",
                "DISTRIBUTION", "ZERO_CROSS_DOWN",
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
            result["Weeks_To_Zero_Cross_Up"] = result.apply(
                lambda row: self._weeks_to_stage(
                    df, row["Start_Date"], row["Boundary_End_Date"], "ZERO_CROSS_UP"
                ), axis=1,
            )
            result["Cycle_Quality"] = result.apply(self._classify_cycle_quality, axis=1)
        return result

    @staticmethod
    def _weeks_to_stage(data, start_date, end_date, stage):
        segment = data.loc[start_date:end_date]
        matches = segment.index[segment["VFI_Cycle_Stage"].eq(stage)]
        return int(segment.index.get_loc(matches[0])) if len(matches) else None

    @staticmethod
    def _classify_cycle_quality(row):
        if row["Cycle_Type"] != "FULL_VFI_CYCLE":
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
        self.validate_data(data)
        df = self.add_forward_returns(data.sort_index())
        stages = tuple(stages or (
            "EARLY_RECOVERY", "SUSTAINED_RECOVERY",
            "APPROACHING_ZERO", "ZERO_CROSS_UP", "ACCUMULATION",
        ))
        previous = df["VFI_Cycle_Stage"].shift(1)
        entry = (
            df["VFI_Cycle_Stage"].isin(stages)
            & (previous.isna() | previous.ne(df["VFI_Cycle_Stage"]))
        )
        events = df.loc[entry].copy()
        if events.empty:
            return events
        events["Cycle_Stage_Event"] = events["VFI_Cycle_Stage"]
        events["Causal_Event"] = True
        return events

    def summarize(self, cycles, complete_only=False):
        if cycles.empty:
            return pd.DataFrame()
        bullish = cycles[cycles["Cycle_Type"] == "FULL_VFI_CYCLE"].copy()
        if complete_only:
            bullish = bullish[bullish["Complete"]].copy()
        if bullish.empty:
            return pd.DataFrame()

        rows = []
        for quality, group in bullish.groupby("Cycle_Quality"):
            row = {
                "Cycle_Quality": quality,
                "Cycles": len(group),
                "Average_Weeks": group["Weeks"].mean(),
                "Average_Recovery_Weeks": group["Recovery_Weeks"].mean(),
                "Average_Peak_Return_Pct": group["Peak_Return_From_Start_Pct"].mean(),
                "Average_Cycle_Return_Pct": group["Price_Return_Pct"].mean(),
                "Median_Cycle_Return_Pct": group["Price_Return_Pct"].median(),
                "Positive_Cycle_Rate_Pct": (group["Price_Return_Pct"] > 0).mean() * 100,
            }
            if "Total_Return_Pct" in group:
                row["Average_Total_Return_Pct"] = group["Total_Return_Pct"].mean()
                row["Median_Total_Return_Pct"] = group["Total_Return_Pct"].median()
            rows.append(row)
        return pd.DataFrame(rows).sort_values(
            "Average_Cycle_Return_Pct", ascending=False
        ).reset_index(drop=True)
