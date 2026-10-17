"""Causal reproduction of TradingView Volume Divergence by MM."""

import numpy as np
import pandas as pd


class VolumeDivergenceMM:
    def __init__(self, vl1=5, vl2=8, lbR=5, lbL=5, range_upper=60, range_lower=5,
                 plot_hidden_bull=False, plot_hidden_bear=False):
        self.plot_hidden_bull = bool(plot_hidden_bull)
        self.plot_hidden_bear = bool(plot_hidden_bear)
        self.vl1 = vl1
        self.vl2 = vl2
        self.vl3 = vl1 + vl2
        self.vl4 = vl2 + self.vl3
        self.vl5 = self.vl3 + self.vl4
        self.lbR = lbR
        self.lbL = lbL
        self.range_upper = range_upper
        self.range_lower = range_lower

    @staticmethod
    def _pine_wma(x, opens, closes, length):
        values = np.asarray(x, dtype=float)
        op = np.asarray(opens, dtype=float)
        cl = np.asarray(closes, dtype=float)
        out = np.full(len(values), np.nan)
        # Input arrays are oldest -> newest, so Pine's WMA weights
        # must increase toward the newest candle (the current bar).
        weights = np.arange(1, length + 1, dtype=float)
        norm = weights.sum()
        for i in range(length - 1, len(values)):
            start = i - length + 1
            w = values[start:i + 1]
            if np.isnan(w).any() or np.isnan(op[start:i + 1]).any() or np.isnan(cl[start:i + 1]).any():
                continue
            factor = np.where(cl[start:i + 1] < op[start:i + 1], -1.0, 1.0)
            out[i] = np.sum(w * weights * factor) / norm
        return pd.Series(out, index=x.index)

    @staticmethod
    def _pivot(values, left, right, high):
        out = np.zeros(len(values), dtype=bool)
        for confirmation in range(left + right, len(values)):
            pivot = confirmation - right
            window = values[pivot-left:pivot+right+1]
            if np.isnan(values[pivot]) or np.isnan(window).any():
                continue
            out[confirmation] = values[pivot] == (np.max(window) if high else np.min(window))
        return out

    def calculate(self, data):
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = data.copy().sort_index()
        v1 = self._pine_wma(df["Volume"], df["Open"], df["Close"], self.vl1)
        v2 = self._pine_wma(v1, df["Open"], df["Close"], self.vl2)
        v3 = self._pine_wma(v2, df["Open"], df["Close"], self.vl3)
        v4 = self._pine_wma(v3, df["Open"], df["Close"], self.vl4)
        df["Volume_Divergence_MM"] = self._pine_wma(v4, df["Open"], df["Close"], self.vl5)

        vol = df["Volume_Divergence_MM"].to_numpy(float)
        low = df["Low"].to_numpy(float)
        high = df["High"].to_numpy(float)
        pl = self._pivot(vol, self.lbL, self.lbR, False)
        ph = self._pivot(vol, self.lbL, self.lbR, True)

        rb = np.zeros(len(df), dtype=bool)
        hb = np.zeros(len(df), dtype=bool)
        rbe = np.zeros(len(df), dtype=bool)
        hbe = np.zeros(len(df), dtype=bool)

        def previous(flags, i):
            hits = np.flatnonzero(flags[:i])
            return None if len(hits) == 0 else int(hits[-1])

        def in_range(flags, i):
            p = previous(flags, i)
            if p is None:
                return False
            bars = i - p - 1
            return self.range_lower <= bars <= self.range_upper

        for i in np.flatnonzero(pl):
            p = i - self.lbR
            q = previous(pl, i)
            if q is None or not in_range(pl, i):
                continue
            q -= self.lbR
            rb[i] = low[p] < low[q] and vol[p] > vol[q]
            hb[i] = low[p] > low[q] and vol[p] < vol[q]

        for i in np.flatnonzero(ph):
            p = i - self.lbR
            q = previous(ph, i)
            if q is None or not in_range(ph, i):
                continue
            q -= self.lbR
            rbe[i] = high[p] > high[q] and vol[p] < vol[q]
            hbe[i] = high[p] < high[q] and vol[p] > vol[q]

        df["Volume_Div_Pivot_Low_Confirmed"] = pl
        df["Volume_Div_Pivot_High_Confirmed"] = ph
        df["Volume_Div_Regular_Bull"] = rb
        df["Volume_Div_Hidden_Bull"] = hb
        df["Volume_Div_Regular_Bear"] = rbe
        df["Volume_Div_Hidden_Bear"] = hbe
        # Raw detections include hidden divergences for research. TradingView's
        # published defaults hide hidden bullish/bearish labels, so the Visible
        # columns respect those defaults unless explicitly enabled.
        df["Volume_Div_Bullish"] = rb | hb
        df["Volume_Div_Bearish"] = rbe | hbe
        visible_bull = rb | (hb if self.plot_hidden_bull else np.zeros(len(df), dtype=bool))
        visible_bear = rbe | (hbe if self.plot_hidden_bear else np.zeros(len(df), dtype=bool))
        df["Volume_Div_Bullish_Visible"] = visible_bull
        df["Volume_Div_Bearish_Visible"] = visible_bear

        signal = np.full(len(df), "", dtype=object)
        signal[rb] = "REGULAR_BULLISH"
        signal[hb] = "HIDDEN_BULLISH"
        signal[rbe] = "REGULAR_BEARISH"
        signal[hbe] = "HIDDEN_BEARISH"
        signal[rb & hb] = "BULLISH_REGULAR_AND_HIDDEN"
        signal[rbe & hbe] = "BEARISH_REGULAR_AND_HIDDEN"
        df["Volume_Div_Signal"] = signal

        visible_signal = np.full(len(df), "", dtype=object)
        visible_signal[rb] = "REGULAR_BULLISH"
        visible_signal[rbe] = "REGULAR_BEARISH"
        if self.plot_hidden_bull:
            visible_signal[hb] = "HIDDEN_BULLISH"
        if self.plot_hidden_bear:
            visible_signal[hbe] = "HIDDEN_BEARISH"
        visible_signal[rb & hb & self.plot_hidden_bull] = "BULLISH_REGULAR_AND_HIDDEN"
        visible_signal[rbe & hbe & self.plot_hidden_bear] = "BEARISH_REGULAR_AND_HIDDEN"
        df["Volume_Div_Visible_Signal"] = visible_signal

        bull_age = []
        bear_age = []
        last_bull = None
        last_bear = None
        for i in range(len(df)):
            if df["Volume_Div_Bullish"].iat[i]:
                last_bull = i
            if df["Volume_Div_Bearish"].iat[i]:
                last_bear = i
            bull_age.append(np.nan if last_bull is None else i - last_bull)
            bear_age.append(np.nan if last_bear is None else i - last_bear)
        df["Volume_Div_Bullish_Age_Bars"] = bull_age
        df["Volume_Div_Bearish_Age_Bars"] = bear_age

        visible_bull_age, visible_bear_age = [], []
        last_visible_bull = last_visible_bear = None
        for i in range(len(df)):
            if visible_bull[i]:
                last_visible_bull = i
            if visible_bear[i]:
                last_visible_bear = i
            visible_bull_age.append(np.nan if last_visible_bull is None else i - last_visible_bull)
            visible_bear_age.append(np.nan if last_visible_bear is None else i - last_visible_bear)
        df["Volume_Div_Bullish_Visible_Age_Bars"] = visible_bull_age
        df["Volume_Div_Bearish_Visible_Age_Bars"] = visible_bear_age
        return df
