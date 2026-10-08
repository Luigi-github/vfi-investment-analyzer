"""Multi-timeframe crypto VFI and Spot/Perpetual confluence research."""
from __future__ import annotations
import numpy as np
import pandas as pd
from app.indicators.vfi import VFIIndicator
from app.analysis.vfi_analyzer import VFIAnalyzer
from app.scoring.vfi_score import VFIScore

TIMEFRAMES = ("1W", "1D", "4H", "1H")

class MultiTimeframeCryptoAnalyzer:
    def __init__(self, vfi_indicator=None):
        self.indicator = vfi_indicator or VFIIndicator()
        self.analyzer = VFIAnalyzer()
        self.scorer = VFIScore()

    def calculate(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        result = {}
        for timeframe, frame in frames.items():
            if frame is None or frame.empty:
                continue
            data = self.indicator.calculate(frame.copy())
            data = self.analyzer.analyze(data)
            data = self.scorer.calculate(data)
            data["VFI_Slope_4B"] = (data["VFI"] - data["VFI"].shift(4)) / 4.0
            data["VFI_Slope_8B"] = (data["VFI"] - data["VFI"].shift(8)) / 8.0
            data["Timeframe"] = timeframe
            result[timeframe] = data
        return result

    @staticmethod
    def _latest(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
        if frame.empty:
            return None
        eligible = frame.loc[frame.index <= timestamp]
        return None if eligible.empty else eligible.iloc[-1]

    @staticmethod
    def _numeric(row: pd.Series, column: str, default=np.nan) -> float:
        value = row.get(column, default)
        return float(value) if pd.notna(value) else float(default)

    def confluence_at(self, calculated: dict[str, pd.DataFrame], timestamp=None) -> dict:
        available = [frame for frame in calculated.values() if not frame.empty]
        if not available:
            return {"Timestamp": timestamp, "Score": 50.0, "Direction": "MIXED", "Alignment": []}
        if timestamp is None:
            timestamp = min(frame.index[-1] for frame in available)
        timestamp = pd.Timestamp(timestamp)
        weights = {"1W": 30, "1D": 30, "4H": 25, "1H": 15}
        positive = {"SUSTAINED_RECOVERY", "APPROACHING_ZERO", "ZERO_CROSS_UP",
                    "ACCUMULATION", "BULLISH_FLOW"}
        negative = {"SUSTAINED_WEAKENING", "DISTRIBUTION", "ZERO_CROSS_DOWN",
                    "BEARISH_FLOW"}
        rows = []
        score = 0.0
        available_weight = 0.0
        for tf in TIMEFRAMES:
            row = self._latest(calculated.get(tf, pd.DataFrame()), timestamp)
            if row is None or pd.isna(row.get("VFI")):
                continue
            stage = row.get("VFI_Cycle_Stage", "NEUTRAL")
            direction = 1 if stage in positive else -1 if stage in negative else 0
            weight = weights[tf]
            score += weight * direction
            available_weight += weight
            rows.append({
                "Timeframe": tf, "Timestamp": row.name, "Stage": stage,
                "VFI": self._numeric(row, "VFI"),
                "Slope_1B": self._numeric(row, "VFI_Slope"),
                "Slope_4B": self._numeric(row, "VFI_Slope_4B"),
                "Slope_8B": self._numeric(row, "VFI_Slope_8B"),
                "Recovery_Score": self._numeric(row, "VFI_Recovery_Score"),
            })
        normalized = 50.0 if available_weight == 0 else 50.0 + 50.0 * score / available_weight
        return {
            "Timestamp": timestamp, "Score": float(normalized),
            "Direction": "BULLISH" if normalized >= 70 else "BEARISH" if normalized <= 30 else "MIXED",
            "Alignment": rows,
        }

    def build_1h_confluence_history(self, calculated: dict[str, pd.DataFrame]) -> pd.DataFrame:
        base = calculated.get("1H")
        if base is None or base.empty:
            return pd.DataFrame()
        out = base[["Close", "VFI", "VFI_Cycle_Stage"]].copy()
        out = out.rename(columns={"VFI": "VFI_1H", "VFI_Cycle_Stage": "Stage_1H"})
        for tf in ("4H", "1D", "1W"):
            frame = calculated.get(tf)
            if frame is None or frame.empty:
                continue
            cols = ["VFI", "VFI_Cycle_Stage", "VFI_Slope", "VFI_Slope_4B", "VFI_Slope_8B"]
            right = frame[cols].copy().rename(columns={
                "VFI": f"VFI_{tf}", "VFI_Cycle_Stage": f"Stage_{tf}",
                "VFI_Slope": f"Slope_1B_{tf}", "VFI_Slope_4B": f"Slope_4B_{tf}",
                "VFI_Slope_8B": f"Slope_8B_{tf}",
            }).reset_index().rename(columns={"Date": "Timestamp"})

            # Binance kline indexes represent candle OPEN time. A higher
            # timeframe VFI is observable only after that candle closes.
            close_delta = {
                "4H": pd.Timedelta(hours=4),
                "1D": pd.Timedelta(days=1),
                "1W": pd.Timedelta(days=7),
            }[tf]
            # Pandas 3.x preserves datetime resolution (ms/us) and
            # merge_asof requires both keys to have exactly the same dtype.
            # Normalize all timestamps to UTC nanoseconds before merging.
            right["Timestamp"] = (
                pd.to_datetime(right["Timestamp"], utc=True)
                + close_delta
            ).astype("datetime64[ns, UTC]")
            right["Signal_Available_At"] = right["Timestamp"]

            left = out.reset_index().rename(columns={"Date": "Timestamp"})
            left["Timestamp"] = pd.to_datetime(
                left["Timestamp"], utc=True
            ).astype("datetime64[ns, UTC]")
            left = left.sort_values("Timestamp")

            out = pd.merge_asof(
                left,
                right.sort_values("Timestamp"),
                on="Timestamp",
                direction="backward",
            ).set_index("Timestamp")
        def stage_score(row):
            weights = {"1W": 30, "1D": 30, "4H": 25, "1H": 15}
            positive = {"SUSTAINED_RECOVERY","APPROACHING_ZERO","ZERO_CROSS_UP","ACCUMULATION","BULLISH_FLOW"}
            negative = {"SUSTAINED_WEAKENING","DISTRIBUTION","ZERO_CROSS_DOWN","BEARISH_FLOW"}
            total = available = 0.0
            for tf,w in weights.items():
                stage = row.get(f"Stage_{tf}")
                if pd.isna(stage): continue
                total += w * (1 if stage in positive else -1 if stage in negative else 0)
                available += w
            return 50.0 if not available else 50.0 + 50.0 * total / available
        out["MTF_Score"] = out.apply(stage_score, axis=1)
        out["MTF_Direction"] = np.select([out["MTF_Score"] >= 70, out["MTF_Score"] <= 30], ["BULLISH", "BEARISH"], default="MIXED")
        return out

    @staticmethod
    def forward_returns(price: pd.Series, horizons=(4, 12, 24, 72)) -> pd.DataFrame:
        result = pd.DataFrame(index=price.index)
        for bars in horizons:
            result[f"Forward_Return_{bars}H"] = price.shift(-bars) / price - 1.0
        return result

    @staticmethod
    def risk_metrics(returns: pd.Series, periods_per_year=8760) -> dict:
        r = pd.Series(returns).dropna()
        if r.empty:
            return {"Observations": 0}
        equity = (1 + r).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else np.nan
        sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 and r.std(ddof=1) else np.nan
        downside = r[r < 0].std(ddof=1)
        sortino = float(r.mean() / downside * np.sqrt(periods_per_year)) if pd.notna(downside) and downside else np.nan
        return {"Observations": int(len(r)), "Cumulative_Return": float(equity.iloc[-1] - 1),
                "Max_Drawdown": float(drawdown.min()), "Annualized_Volatility": ann_vol,
                "Sharpe": sharpe, "Sortino": sortino, "Positive_Rate": float((r > 0).mean())}
