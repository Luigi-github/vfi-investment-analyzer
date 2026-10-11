from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.binance_market_data import BinanceMarketDataService
from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM

# Crypto universe is discovered dynamically from the top-200 liquid Binance spot/perpetual intersection.
# Historical non-crypto examples supplied for the earlier VFI cycle research.
YAHOO_ASSETS = {
    "DDD": "DDD",
    "INTC": "INTC",
    "EEM": "EEM",
    "NVDA": "NVDA",
    "SPY": "SPY",
    "AMD": "AMD",
    "MINEROS": "MINEROS.CL",
    "ECOPETROL_BVC": "ECOPETROL.CL",
    "ECOPETROL_ADR": "EC",
    "ISA": "ISA.CL",
    "SQM": "SQM",
    "NU": "NU",
    "AMZN": "AMZN",
    "PFCIBEST": "PFCIBEST.CL",
    "PFDAVVNDA": "PFDAVVNDA.CL",
    "PFAVAL": "PFAVAL.CL",
    "GEB": "GEB.CL",
    "SQM_B": "SQM-B.SN",
}

HORIZONS_DAYS = (10, 20, 40, 60)
CRYPTO_UNIVERSE_SIZE = 200
MIN_DAILY_BARS = 260
MIN_WEEKLY_BARS_FOR_DIVERGENCE = 100
WEEKLY_DIVERGENCE_FRESH_DAYS = 35
DAILY_DIVERGENCE_FRESH_DAYS = 20
COOLDOWN_DAILY_BARS = 20
VFI_MAX_ESTIMATED_DAYS_TO_ZERO = 20.0
DIVERGENCE_MAX_ESTIMATED_WEEKS_TO_ZERO = 8.0
ZERO_CROSS_MAX_NORMALIZED_DISTANCE = 0.25
STRONG_SLOPE_MIN_NORMALIZED_CHANGE = 0.20


def discover_crypto_universe(service: BinanceMarketDataService, limit: int = CRYPTO_UNIVERSE_SIZE) -> list[dict]:
    """Return liquid Binance spot/perpetual intersections, ranked by both markets' 24h quote volume."""
    spot = service.discover_spot_usdt(0)
    perp = service.discover_usdt_perpetuals(0)
    spot_rank = {symbol: i + 1 for i, symbol in enumerate(spot)}
    perp_rank = {symbol: i + 1 for i, symbol in enumerate(perp)}
    common = set(spot_rank) & set(perp_rank)
    if len(common) < limit:
        raise RuntimeError(f"Only {len(common)} Binance USDT spot/perpetual intersections found; need {limit}")

    spot_tickers = service._get(service.SPOT_BASE, "/api/v3/ticker/24hr", {})
    perp_tickers = service._get(service.FUTURES_BASE, "/fapi/v1/ticker/24hr", {})
    spot_by_symbol = {row["symbol"]: row for row in spot_tickers if row.get("symbol") in common}
    perp_by_symbol = {row["symbol"]: row for row in perp_tickers if row.get("symbol") in common}
    records = []
    for symbol in common:
        srow, prow = spot_by_symbol.get(symbol), perp_by_symbol.get(symbol)
        if srow is None or prow is None:
            continue
        sr, pr = spot_rank[symbol], perp_rank[symbol]
        records.append({
            "Symbol": symbol,
            "Spot_Volume_Rank": sr,
            "Perp_Volume_Rank": pr,
            "Spot_24h_Quote_Volume_USDT": float(srow.get("quoteVolume", 0) or 0),
            "Perp_24h_Quote_Volume_USDT": float(prow.get("quoteVolume", 0) or 0),
            "Combined_Volume_Rank_Sum": sr + pr,
            "Worst_Market_Volume_Rank": max(sr, pr),
        })
    records.sort(key=lambda row: (
        row["Combined_Volume_Rank_Sum"], row["Worst_Market_Volume_Rank"], row["Symbol"]
    ))
    records = records[:limit]
    if len(records) < limit:
        raise RuntimeError(f"Only {len(records)} complete liquidity records found; need {limit}")
    for rank, record in enumerate(records, 1):
        record["Universe_Rank"] = rank
    return records


def closed_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude today's possibly incomplete daily candle."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy().sort_index()
    idx = pd.DatetimeIndex(data.index)
    if idx.tz is not None:
        today = pd.Timestamp.now(tz="UTC").normalize()
        data = data[idx.normalize() < today]
    else:
        today = pd.Timestamp.now().normalize()
        data = data[idx.normalize() < today]
    data = data[~data.index.duplicated(keep="last")]
    return data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def to_weekly(daily: pd.DataFrame, crypto: bool) -> pd.DataFrame:
    # TradingView crypto weeks normally close Sunday; exchange-traded assets
    # use Friday. Only fully completed weekly candles are retained.
    rule = "W-SUN" if crypto else "W-FRI"
    weekly = daily.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if daily.empty:
        return weekly.iloc[0:0]
    last_daily_date = pd.Timestamp(daily.index[-1]).normalize()
    return weekly[pd.DatetimeIndex(weekly.index).normalize() <= last_daily_date]


def add_vfi_features(daily: pd.DataFrame, indicator: VFIIndicator) -> pd.DataFrame:
    data = indicator.calculate(daily)
    vfi = data["VFI"].astype(float)
    # Normalize slopes by trailing VFI volatility so "strong slope" is
    # comparable across assets whose raw VFI scales differ.
    scale = vfi.rolling(252, min_periods=60).std().replace(0, np.nan)
    data["VFI_Z_252"] = vfi / scale
    for days in (5, 10, 20):
        data[f"VFI_Slope_{days}D"] = (vfi - vfi.shift(days)) / days
        data[f"VFI_Slope_{days}D_Normalized"] = (vfi - vfi.shift(days)) / scale
    s5 = data["VFI_Slope_5D"]
    s10 = data["VFI_Slope_10D"]
    s20 = data["VFI_Slope_20D"]
    z = data["VFI_Z_252"]
    data["VFI_Negative_Near_Zero"] = (vfi < 0) & (z >= -0.75)
    data["VFI_Positive_Slopes"] = (s5 > 0) & (s10 > 0) & (s20 > 0)
    data["VFI_Strong_Rising_Slope"] = (
        data["VFI_Slope_10D_Normalized"].ge(STRONG_SLOPE_MIN_NORMALIZED_CHANGE)
        & data["VFI_Slope_20D_Normalized"].gt(0)
        & data["VFI_Positive_Slopes"]
    )
    # Estimate time to zero using the raw 10-day slope (VFI units per day).
    data["VFI_Estimated_Days_To_Zero"] = np.where(
        (vfi < 0) & (s10 > 0), -vfi / s10, np.nan
    )
    data["VFI_Early_Setup"] = (
        data["VFI_Negative_Near_Zero"]
        & data["VFI_Strong_Rising_Slope"]
        & data["VFI_Estimated_Days_To_Zero"].between(0, VFI_MAX_ESTIMATED_DAYS_TO_ZERO)
    )
    data["VFI_Zero_Cross_Up"] = (vfi.shift(1) < 0) & (vfi >= 0) & (s5 > 0)
    data["VFI_Zero_Cross_Up_Near_Zero"] = (
        data["VFI_Zero_Cross_Up"] & z.le(ZERO_CROSS_MAX_NORMALIZED_DISTANCE)
    )
    data["VFI_Zero_Cross_Up_Recent_Near_Zero"] = (
        data["VFI_Zero_Cross_Up_Near_Zero"].rolling(4, min_periods=1).max().astype(bool)
        & z.le(ZERO_CROSS_MAX_NORMALIZED_DISTANCE)
        & data["VFI_Positive_Slopes"]
        & data["VFI_Strong_Rising_Slope"]
    )
    data["VFI_Recovery_From_Negative"] = (vfi < 0) & (s5 > 0) & (s10 > 0)
    return data


def add_divergence_cycle_features(
    frame: pd.DataFrame, periods=(1, 2, 4), suffix="W",
    scale_window=52, min_periods=20,
) -> pd.DataFrame:
    """Measure whether the divergence oscillator itself is rising toward/crossing zero."""
    data = frame.copy()
    osc = data["Volume_Divergence_MM"].astype(float)
    scale = osc.rolling(scale_window, min_periods=min_periods).std().replace(0, np.nan)
    data[f"Div_Osc_Z_{suffix}"] = osc / scale
    for bars in periods:
        data[f"Div_Osc_Slope_{bars}{suffix}"] = (osc - osc.shift(bars)) / bars
        data[f"Div_Osc_Slope_{bars}{suffix}_Normalized"] = (osc - osc.shift(bars)) / scale
    slope_cols = [f"Div_Osc_Slope_{bars}{suffix}" for bars in periods]
    data[f"Div_Osc_Negative_Near_Zero_{suffix}"] = (osc < 0) & data[f"Div_Osc_Z_{suffix}"].ge(-0.75)
    data[f"Div_Osc_Positive_Slopes_{suffix}"] = data[slope_cols].gt(0).all(axis=1)
    fast_norm = data[f"Div_Osc_Slope_{periods[1]}{suffix}_Normalized"]
    slow_norm = data[f"Div_Osc_Slope_{periods[2]}{suffix}_Normalized"]
    data[f"Div_Osc_Strong_Rising_Slope_{suffix}"] = (
        fast_norm.ge(STRONG_SLOPE_MIN_NORMALIZED_CHANGE)
        & slow_norm.gt(0)
        & data[f"Div_Osc_Positive_Slopes_{suffix}"]
    )
    raw_fast_slope = data[f"Div_Osc_Slope_{periods[1]}{suffix}"]
    data[f"Div_Osc_Estimated_Weeks_To_Zero_{suffix}"] = np.where(
        (osc < 0) & (raw_fast_slope > 0), -osc / raw_fast_slope, np.nan
    )
    data[f"Div_Osc_Early_Setup_{suffix}"] = (
        data[f"Div_Osc_Negative_Near_Zero_{suffix}"]
        & data[f"Div_Osc_Strong_Rising_Slope_{suffix}"]
        & data[f"Div_Osc_Estimated_Weeks_To_Zero_{suffix}"].between(
            0, DIVERGENCE_MAX_ESTIMATED_WEEKS_TO_ZERO
        )
    )
    data[f"Div_Osc_Zero_Cross_Up_{suffix}"] = (
        (osc.shift(1) < 0) & (osc >= 0)
        & data[slope_cols[0]].gt(0)
        & data[f"Div_Osc_Z_{suffix}"].le(ZERO_CROSS_MAX_NORMALIZED_DISTANCE)
    )
    data[f"Div_Osc_Zero_Cross_Up_Recent_{suffix}"] = (
        data[f"Div_Osc_Zero_Cross_Up_{suffix}"].rolling(3, min_periods=1).max().astype(bool)
        & data[f"Div_Osc_Z_{suffix}"].le(ZERO_CROSS_MAX_NORMALIZED_DISTANCE)
        & data[f"Div_Osc_Positive_Slopes_{suffix}"]
        & data[f"Div_Osc_Strong_Rising_Slope_{suffix}"]
    )
    return data


def fresh_signal(signal: pd.Series, days: int) -> pd.Series:
    """True only while the latest confirmed signal is still recent."""
    signal = signal.fillna(False).astype(bool)
    last = pd.Series(np.where(signal, np.arange(len(signal)), np.nan), index=signal.index).ffill()
    age = pd.Series(np.arange(len(signal)), index=signal.index) - last
    return age.le(days).fillna(False)


def cooldown_positions(mask: pd.Series, cooldown: int = COOLDOWN_DAILY_BARS) -> list[int]:
    positions = np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool))
    selected = []
    last = -10**9
    for pos in positions:
        if pos - last >= cooldown:
            selected.append(int(pos))
            last = int(pos)
    return selected


def get_crypto_daily(service: BinanceMarketDataService, symbol: str) -> pd.DataFrame:
    frame = service.get_klines(symbol, interval="1d", market="spot", max_bars=1200)
    if frame.empty:
        raise ValueError("Binance returned no daily spot candles")
    return frame[["Open", "High", "Low", "Close", "Volume"]].copy()


def classify_two_stage_opportunities(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the user's ordered screen: daily VFI first, completed-week divergence second."""
    joined = frame.copy()
    joined["VFI_Stage"] = np.select(
        [
            joined["VFI_Early_Setup"].fillna(False),
            joined["VFI_Zero_Cross_Up_Recent_Near_Zero"].fillna(False),
            joined["VFI"].ge(0),
            joined["VFI"].lt(0),
        ],
        ["PRE_ZERO_SETUP", "FRESH_ZERO_CROSS", "POSITIVE_AFTER_CROSS_OR_LATE", "NEGATIVE_NOT_READY"],
        default="NO_DATA",
    )
    joined["VFI_Opportunity_Ready"] = joined["VFI_Stage"].isin(["PRE_ZERO_SETUP", "FRESH_ZERO_CROSS"])
    joined["Divergence_Cycle_Stage"] = np.select(
        [
            ~joined["Weekly_Divergence_History_Sufficient"].fillna(False),
            joined["Weekly_Div_Osc_Early_Setup"].fillna(False),
            joined["Weekly_Div_Osc_Zero_Cross_Up_Recent"].fillna(False),
            joined["Weekly_Div_Value"].ge(0),
            joined["Weekly_Div_Value"].lt(0),
        ],
        ["NO_HISTORY", "PRE_ZERO_SETUP", "FRESH_ZERO_CROSS", "POSITIVE_AFTER_CROSS_OR_LATE", "NEGATIVE_NOT_READY"],
        default="NO_HISTORY",
    )
    joined["Divergence_Cycle_Ready"] = joined["Divergence_Cycle_Stage"].isin(["PRE_ZERO_SETUP", "FRESH_ZERO_CROSS"])
    joined["Two_Stage_Opportunity"] = joined["VFI_Opportunity_Ready"] & joined["Divergence_Cycle_Ready"]
    joined["Opportunity_Tier"] = np.select(
        [
            joined["VFI_Stage"].eq("PRE_ZERO_SETUP") & joined["Divergence_Cycle_Stage"].eq("PRE_ZERO_SETUP"),
            joined["VFI_Stage"].eq("PRE_ZERO_SETUP") & joined["Divergence_Cycle_Stage"].eq("FRESH_ZERO_CROSS"),
            joined["VFI_Stage"].eq("FRESH_ZERO_CROSS") & joined["Divergence_Cycle_Stage"].eq("PRE_ZERO_SETUP"),
            joined["VFI_Stage"].eq("FRESH_ZERO_CROSS") & joined["Divergence_Cycle_Stage"].eq("FRESH_ZERO_CROSS"),
            joined["VFI_Opportunity_Ready"],
            joined["Divergence_Cycle_Ready"],
        ],
        [1, 2, 3, 4, 5, 6],
        default=7,
    ).astype(int)
    joined["Candidate_Status"] = np.select(
        [
            joined["Two_Stage_Opportunity"],
            joined["VFI_Opportunity_Ready"],
            joined["Divergence_Cycle_Ready"],
        ],
        ["TWO_STAGE_CANDIDATE", "VFI_READY_WAITING_FOR_DIVERGENCE", "DIVERGENCE_READY_WAITING_FOR_VFI"],
        default="NO_EARLY_SETUP",
    )
    joined["Setup_Score"] = (
        pd.to_numeric(joined["VFI_Slope_10D_Normalized"], errors="coerce").clip(lower=0).fillna(0)
        + pd.to_numeric(joined["Weekly_Div_Osc_Slope_2W_Normalized"], errors="coerce").clip(lower=0).fillna(0)
        + joined["Weekly_Bullish_Div_Recent"].fillna(False).astype(float) * 0.25
    )
    joined["Selection_Reason"] = np.select(
        [
            joined["Two_Stage_Opportunity"] & joined["VFI_Stage"].eq("PRE_ZERO_SETUP") & joined["Divergence_Cycle_Stage"].eq("PRE_ZERO_SETUP"),
            joined["Two_Stage_Opportunity"] & joined["VFI_Stage"].eq("PRE_ZERO_SETUP") & joined["Divergence_Cycle_Stage"].eq("FRESH_ZERO_CROSS"),
            joined["Two_Stage_Opportunity"] & joined["VFI_Stage"].eq("FRESH_ZERO_CROSS") & joined["Divergence_Cycle_Stage"].eq("PRE_ZERO_SETUP"),
            joined["Two_Stage_Opportunity"] & joined["VFI_Stage"].eq("FRESH_ZERO_CROSS") & joined["Divergence_Cycle_Stage"].eq("FRESH_ZERO_CROSS"),
            joined["VFI_Opportunity_Ready"],
            joined["Divergence_Cycle_Ready"],
        ],
        [
            "Both oscillators negative, rising strongly, and projected to cross zero soon",
            "VFI is still in early negative recovery; divergence oscillator has just crossed zero",
            "VFI has just crossed zero; divergence oscillator is still in early negative recovery",
            "Both oscillators have freshly crossed zero and remain close to it",
            "Stage 1 passed: waiting for weekly divergence oscillator confirmation",
            "Stage 2 is promising, but VFI Stage 1 has not passed; not an investment candidate yet",
        ],
        default="No qualifying early-cycle setup",
    )
    return joined


def build_asset_events(asset: str, daily_raw: pd.DataFrame, crypto: bool,
                       vfi_indicator: VFIIndicator, divergence: VolumeDivergenceMM):
    daily = closed_daily(daily_raw)
    if len(daily) < MIN_DAILY_BARS:
        raise ValueError(f"Only {len(daily)} closed daily bars; need at least {MIN_DAILY_BARS}")
    daily_vfi = add_vfi_features(daily, vfi_indicator)
    daily_div = divergence.calculate(daily)
    weekly = to_weekly(daily, crypto)
    weekly_history_sufficient = len(weekly) >= MIN_WEEKLY_BARS_FOR_DIVERGENCE
    weekly_div = divergence.calculate(weekly)
    weekly_div = add_divergence_cycle_features(
        weekly_div, periods=(1, 2, 4), suffix="W", scale_window=52, min_periods=20
    )

    # A chained WMA warm-up plus confirmed pivots needs a long weekly sample.
    # Keep VFI-only daily results for younger assets, but do not treat missing
    # weekly-divergence history as a negative or valid divergence reading.
    wbull = weekly_div["Volume_Div_Bullish"].fillna(False).astype(bool)
    wbear = weekly_div["Volume_Div_Bearish"].fillna(False).astype(bool)
    weekly_signal_names = weekly_div["Volume_Div_Signal"].astype(str)
    weekly_bull_age = weekly_div["Volume_Div_Bullish_Age_Bars"].copy()
    weekly_bear_age = weekly_div["Volume_Div_Bearish_Age_Bars"].copy()
    if not weekly_history_sufficient:
        wbull = pd.Series(False, index=weekly.index)
        wbear = pd.Series(False, index=weekly.index)
        weekly_signal_names = pd.Series("", index=weekly.index, dtype=object)
        weekly_bull_age = pd.Series(np.nan, index=weekly.index)
        weekly_bear_age = pd.Series(np.nan, index=weekly.index)
        weekly_div["Div_Osc_Early_Setup_W"] = False
        weekly_div["Div_Osc_Zero_Cross_Up_W"] = False
        weekly_div["Div_Osc_Zero_Cross_Up_Recent_W"] = False
        weekly_div["Div_Osc_Negative_Near_Zero_W"] = False
        weekly_div["Div_Osc_Positive_Slopes_W"] = False
        weekly_div["Div_Osc_Strong_Rising_Slope_W"] = False
        weekly_div["Div_Osc_Estimated_Weeks_To_Zero_W"] = np.nan
    weekly_signal = pd.DataFrame({
        "Weekly_Bullish_Div_Confirmed": wbull,
        "Weekly_Bearish_Div_Confirmed": wbear,
        "Weekly_Bullish_Div_Age_Bars": weekly_bull_age,
        "Weekly_Bearish_Div_Age_Bars": weekly_bear_age,
        "Weekly_Last_Bullish_Signal": weekly_signal_names.where(wbull, np.nan).ffill(),
        "Weekly_Last_Bearish_Signal": weekly_signal_names.where(wbear, np.nan).ffill(),
        "Weekly_Div_Signal": weekly_signal_names,
        "Weekly_Div_Value": weekly_div["Volume_Divergence_MM"],
        "Weekly_Div_Osc_Z_52": weekly_div["Div_Osc_Z_W"],
        "Weekly_Div_Osc_Slope_1W": weekly_div["Div_Osc_Slope_1W"],
        "Weekly_Div_Osc_Slope_2W_Normalized": weekly_div["Div_Osc_Slope_2W_Normalized"],
        "Weekly_Div_Osc_Slope_4W_Normalized": weekly_div["Div_Osc_Slope_4W_Normalized"],
        "Weekly_Div_Osc_Estimated_Weeks_To_Zero": weekly_div["Div_Osc_Estimated_Weeks_To_Zero_W"],
        "Weekly_Div_Osc_Negative_Near_Zero": weekly_div["Div_Osc_Negative_Near_Zero_W"],
        "Weekly_Div_Osc_Positive_Slopes": weekly_div["Div_Osc_Positive_Slopes_W"],
        "Weekly_Div_Osc_Strong_Rising_Slope": weekly_div["Div_Osc_Strong_Rising_Slope_W"],
        "Weekly_Div_Osc_Early_Setup": weekly_div["Div_Osc_Early_Setup_W"],
        "Weekly_Div_Osc_Zero_Cross_Up": weekly_div["Div_Osc_Zero_Cross_Up_W"],
        "Weekly_Div_Osc_Zero_Cross_Up_Recent": weekly_div["Div_Osc_Zero_Cross_Up_Recent_W"],
    }).reset_index().rename(columns={weekly_div.index.name or "index": "Weekly_Date"})
    weekly_signal["Weekly_Date"] = pd.to_datetime(weekly_signal["Weekly_Date"])
    left = daily_vfi.reset_index().rename(columns={daily_vfi.index.name or "index": "Date"})
    left["Date"] = pd.to_datetime(left["Date"])
    left = left.sort_values("Date")
    weekly_signal = weekly_signal.sort_values("Weekly_Date")
    joined = pd.merge_asof(
        left, weekly_signal, left_on="Date", right_on="Weekly_Date",
        direction="backward", allow_exact_matches=True,
    ).set_index("Date")
    # The as-of join carries the last completed weekly row across daily
    # candles. Use the indicator's actual age-in-weeks, not the carried
    # boolean, or a confirmed signal would incorrectly reset its age daily.
    joined["Weekly_Bullish_Div_Age_Bars"] = pd.to_numeric(
        joined["Weekly_Bullish_Div_Age_Bars"], errors="coerce"
    )
    joined["Weekly_Bearish_Div_Age_Bars"] = pd.to_numeric(
        joined["Weekly_Bearish_Div_Age_Bars"], errors="coerce"
    )
    max_weekly_age = WEEKLY_DIVERGENCE_FRESH_DAYS / 7.0
    joined["Weekly_Bullish_Div_Recent"] = (
        joined["Weekly_Bullish_Div_Age_Bars"].le(max_weekly_age)
    )
    joined["Weekly_Bearish_Div_Recent"] = (
        joined["Weekly_Bearish_Div_Age_Bars"].le(max_weekly_age)
    )

    # Daily divergence is included as a comparison, but all signals are daily
    # or weekly; no intraday timeframe enters this study.
    joined["Daily_Bullish_Div_Recent"] = fresh_signal(
        daily_div["Volume_Div_Bullish"].reindex(joined.index, fill_value=False),
        DAILY_DIVERGENCE_FRESH_DAYS,
    )
    joined["Daily_Bearish_Div_Recent"] = fresh_signal(
        daily_div["Volume_Div_Bearish"].reindex(joined.index, fill_value=False),
        DAILY_DIVERGENCE_FRESH_DAYS,
    )
    daily_signal_names = daily_div["Volume_Div_Signal"].astype(str)
    daily_bull_age = daily_div["Volume_Div_Bullish_Age_Bars"].reindex(joined.index)
    daily_bear_age = daily_div["Volume_Div_Bearish_Age_Bars"].reindex(joined.index)
    daily_age = pd.concat([daily_bull_age, daily_bear_age], axis=1).min(axis=1, skipna=True)
    daily_bull_signal = daily_signal_names.where(daily_div["Volume_Div_Bullish"].fillna(False)).ffill().reindex(joined.index)
    daily_bear_signal = daily_signal_names.where(daily_div["Volume_Div_Bearish"].fillna(False)).ffill().reindex(joined.index)
    daily_bull_is_newer = daily_bull_age.fillna(np.inf).le(daily_bear_age.fillna(np.inf))
    joined["Daily_Last_Confirmed_Div_Signal"] = np.where(daily_bull_is_newer, daily_bull_signal, daily_bear_signal)
    joined["Daily_Last_Confirmed_Div_Signal"] = joined["Daily_Last_Confirmed_Div_Signal"].fillna("")
    joined["Daily_Div_Age_Bars"] = daily_age
    joined["Daily_Div_Signal"] = np.where(
        daily_age.le(DAILY_DIVERGENCE_FRESH_DAYS), joined["Daily_Last_Confirmed_Div_Signal"], "NONE_RECENT"
    )
    bull_week_age = joined["Weekly_Bullish_Div_Age_Bars"]
    bear_week_age = joined["Weekly_Bearish_Div_Age_Bars"]
    bull_is_newer = bull_week_age.fillna(np.inf).le(bear_week_age.fillna(np.inf))
    most_recent_week_age = pd.concat([bull_week_age, bear_week_age], axis=1).min(axis=1, skipna=True)
    joined["Weekly_Div_Age_Days"] = most_recent_week_age * 7.0
    joined["Weekly_Last_Confirmed_Div_Signal"] = np.where(
        bull_is_newer, joined["Weekly_Last_Bullish_Signal"], joined["Weekly_Last_Bearish_Signal"]
    )
    joined["Weekly_Last_Confirmed_Div_Signal"] = joined["Weekly_Last_Confirmed_Div_Signal"].fillna("")
    joined["Weekly_Div_Signal"] = np.where(
        most_recent_week_age.le(max_weekly_age), joined["Weekly_Last_Confirmed_Div_Signal"], "NONE_RECENT"
    )
    joined["Weekly_Div_Signal"] = np.where(
        weekly_history_sufficient, joined["Weekly_Div_Signal"], "INSUFFICIENT_WEEKLY_HISTORY"
    )
    joined["Weekly_Divergence_History_Sufficient"] = weekly_history_sufficient
    joined = classify_two_stage_opportunities(joined)

    conditions = {
        "VFI_EARLY_SETUP": joined["VFI_Early_Setup"],
        "VFI_ZERO_CROSS_UP": joined["VFI_Zero_Cross_Up"],
        "WEEKLY_BULLISH_DIVERGENCE": joined["Weekly_Bullish_Div_Recent"],
        "VFI_SETUP_PLUS_WEEKLY_BULL_DIV": joined["VFI_Early_Setup"] & joined["Weekly_Bullish_Div_Recent"],
        "VFI_SETUP_PLUS_DAILY_BULL_DIV": joined["VFI_Early_Setup"] & joined["Daily_Bullish_Div_Recent"],
        "VFI_RECOVERY_PLUS_WEEKLY_BULL_DIV": joined["VFI_Recovery_From_Negative"] & joined["Weekly_Bullish_Div_Recent"],
        "VFI_CROSS_PLUS_WEEKLY_BULL_DIV": joined["VFI_Zero_Cross_Up"] & joined["Weekly_Bullish_Div_Recent"],
        "WEEKLY_DIV_OSC_EARLY_SETUP": joined["Weekly_Div_Osc_Early_Setup"],
        "VFI_SETUP_PLUS_WEEKLY_DIV_OSC_SETUP": joined["VFI_Early_Setup"] & joined["Weekly_Div_Osc_Early_Setup"],
        "VFI_SETUP_PLUS_WEEKLY_DIV_OSC_SETUP_AND_BULL_DIV": joined["VFI_Early_Setup"] & joined["Weekly_Div_Osc_Early_Setup"] & joined["Weekly_Bullish_Div_Recent"],
        "VFI_SETUP_PLUS_WEEKLY_DIV_OSC_ZERO_CROSS": joined["VFI_Early_Setup"] & joined["Weekly_Div_Osc_Zero_Cross_Up"],
        "VFI_SETUP_PLUS_WEEKLY_DIV_OSC_RECENT_CROSS": joined["VFI_Early_Setup"] & joined["Weekly_Div_Osc_Zero_Cross_Up_Recent"],
        "VFI_RECENT_CROSS_PLUS_WEEKLY_DIV_OSC_SETUP": joined["VFI_Zero_Cross_Up_Recent_Near_Zero"] & joined["Weekly_Div_Osc_Early_Setup"],
        "BOTH_OSCILLATORS_FRESH_ZERO_CROSS": joined["VFI_Zero_Cross_Up_Recent_Near_Zero"] & joined["Weekly_Div_Osc_Zero_Cross_Up_Recent"],
        "VFI_SETUP_PLUS_WEEKLY_DIV_OSC_CROSS_AND_BULL_DIV": joined["VFI_Early_Setup"] & joined["Weekly_Div_Osc_Zero_Cross_Up_Recent"] & joined["Weekly_Bullish_Div_Recent"],
    }

    events = []
    for strategy, mask in conditions.items():
        onset = mask.fillna(False) & ~mask.fillna(False).shift(1, fill_value=False)
        for pos in cooldown_positions(onset):
            row = joined.iloc[pos]
            record = {
                "Asset": asset,
                "Source": "BINANCE_SPOT" if crypto else "YAHOO_FINANCE",
                "Signal_Date": joined.index[pos].strftime("%Y-%m-%d"),
                "Strategy": strategy,
                "Price": float(row["Close"]),
                "VFI_Daily": float(row["VFI"]) if pd.notna(row["VFI"]) else np.nan,
                "VFI_Z_252": float(row["VFI_Z_252"]) if pd.notna(row["VFI_Z_252"]) else np.nan,
                "VFI_Slope_5D": float(row["VFI_Slope_5D"]) if pd.notna(row["VFI_Slope_5D"]) else np.nan,
                "VFI_Slope_10D_Normalized": float(row["VFI_Slope_10D_Normalized"]) if pd.notna(row["VFI_Slope_10D_Normalized"]) else np.nan,
                "VFI_Slope_20D_Normalized": float(row["VFI_Slope_20D_Normalized"]) if pd.notna(row["VFI_Slope_20D_Normalized"]) else np.nan,
                "Weekly_Div_Signal": str(row.get("Weekly_Div_Signal", "")),
                "Weekly_Last_Confirmed_Div_Signal": str(row.get("Weekly_Last_Confirmed_Div_Signal", "")),
                "Weekly_Div_Age_Days": float(row["Weekly_Div_Age_Days"]) if pd.notna(row["Weekly_Div_Age_Days"]) else np.nan,
                "Weekly_Divergence_History_Sufficient": bool(row["Weekly_Divergence_History_Sufficient"]),
                "Weekly_Div_Oscillator": float(row["Weekly_Div_Value"]) if pd.notna(row["Weekly_Div_Value"]) else np.nan,
                "Weekly_Div_Osc_Z_52": float(row["Weekly_Div_Osc_Z_52"]) if pd.notna(row["Weekly_Div_Osc_Z_52"]) else np.nan,
                "Weekly_Div_Osc_Slope_1W": float(row["Weekly_Div_Osc_Slope_1W"]) if pd.notna(row["Weekly_Div_Osc_Slope_1W"]) else np.nan,
                "Weekly_Div_Osc_Slope_2W_Normalized": float(row["Weekly_Div_Osc_Slope_2W_Normalized"]) if pd.notna(row["Weekly_Div_Osc_Slope_2W_Normalized"]) else np.nan,
                "Weekly_Div_Osc_Slope_4W_Normalized": float(row["Weekly_Div_Osc_Slope_4W_Normalized"]) if pd.notna(row["Weekly_Div_Osc_Slope_4W_Normalized"]) else np.nan,
                "Weekly_Div_Osc_Negative_Near_Zero": bool(row["Weekly_Div_Osc_Negative_Near_Zero"]),
                "Weekly_Div_Osc_Positive_Slopes": bool(row["Weekly_Div_Osc_Positive_Slopes"]),
                "Weekly_Div_Osc_Strong_Rising_Slope": bool(row["Weekly_Div_Osc_Strong_Rising_Slope"]),
                "Weekly_Div_Osc_Early_Setup": bool(row["Weekly_Div_Osc_Early_Setup"]),
                "Weekly_Div_Osc_Zero_Cross_Up": bool(row["Weekly_Div_Osc_Zero_Cross_Up"]),
                "Weekly_Div_Osc_Zero_Cross_Up_Recent": bool(row["Weekly_Div_Osc_Zero_Cross_Up_Recent"]),
                "Weekly_Div_Osc_Estimated_Weeks_To_Zero": float(row["Weekly_Div_Osc_Estimated_Weeks_To_Zero"]) if pd.notna(row["Weekly_Div_Osc_Estimated_Weeks_To_Zero"]) else np.nan,
                "Daily_Div_Signal": str(row.get("Daily_Div_Signal", "")),
                "Daily_Div_Age_Bars": float(row["Daily_Div_Age_Bars"]) if pd.notna(row["Daily_Div_Age_Bars"]) else np.nan,
                "VFI_Negative_Near_Zero": bool(row["VFI_Negative_Near_Zero"]),
                "VFI_Positive_Slopes": bool(row["VFI_Positive_Slopes"]),
                "VFI_Strong_Rising_Slope": bool(row["VFI_Strong_Rising_Slope"]),
                "VFI_Early_Setup": bool(row["VFI_Early_Setup"]),
                "VFI_Zero_Cross_Up": bool(row["VFI_Zero_Cross_Up"]),
                "VFI_Zero_Cross_Up_Recent_Near_Zero": bool(row["VFI_Zero_Cross_Up_Recent_Near_Zero"]),
                "VFI_Estimated_Days_To_Zero": float(row["VFI_Estimated_Days_To_Zero"]) if pd.notna(row["VFI_Estimated_Days_To_Zero"]) else np.nan,
                "VFI_Stage": str(row["VFI_Stage"]),
                "Divergence_Cycle_Stage": str(row["Divergence_Cycle_Stage"]),
                "VFI_Opportunity_Ready": bool(row["VFI_Opportunity_Ready"]),
                "Divergence_Cycle_Ready": bool(row["Divergence_Cycle_Ready"]),
                "Two_Stage_Opportunity": bool(row["Two_Stage_Opportunity"]),
                "Opportunity_Tier": int(row["Opportunity_Tier"]),
                "Candidate_Status": str(row["Candidate_Status"]),
                "Selection_Reason": str(row["Selection_Reason"]),
                "Setup_Score": float(row["Setup_Score"]),
                "Weekly_Bullish_Div_Recent": bool(row["Weekly_Bullish_Div_Recent"]),
                "Daily_Bullish_Div_Recent": bool(row["Daily_Bullish_Div_Recent"]),
                "Bars_History": len(joined),
            }
            for horizon in HORIZONS_DAYS:
                future_pos = pos + horizon
                record[f"Return_{horizon}D"] = (
                    float(joined["Close"].iloc[future_pos] / row["Close"] - 1)
                    if future_pos < len(joined) else np.nan
                )
            events.append(record)

    latest = joined.iloc[-1]
    weekly_age = latest["Weekly_Div_Age_Days"]
    signal_type = str(latest["Candidate_Status"])
    if bool(latest["Two_Stage_Opportunity"]) and bool(latest["Weekly_Bullish_Div_Recent"]):
        signal_type = "TWO_STAGE_CANDIDATE + RECENT_BULLISH_DIVERGENCE"
    elif bool(latest["Two_Stage_Opportunity"]):
        signal_type = "TWO_STAGE_CANDIDATE"
    elif str(latest["VFI_Stage"]) == "POSITIVE_AFTER_CROSS_OR_LATE":
        signal_type = "VFI_ALREADY_ABOVE_ZERO; EXCLUDED_FROM_EARLY_ENTRY"
    elif str(latest["Divergence_Cycle_Stage"]) == "POSITIVE_AFTER_CROSS_OR_LATE":
        signal_type = "DIVERGENCE_OSC_ALREADY_ABOVE_ZERO; EXCLUDED_FROM_EARLY_ENTRY"
    current = {
        "Asset": asset,
        "Source": "BINANCE_SPOT" if crypto else "YAHOO_FINANCE",
        "Date": joined.index[-1].strftime("%Y-%m-%d"),
        "Price": float(latest["Close"]),
        "VFI_Daily": float(latest["VFI"]) if pd.notna(latest["VFI"]) else np.nan,
        "VFI_Z_252": float(latest["VFI_Z_252"]) if pd.notna(latest["VFI_Z_252"]) else np.nan,
        "VFI_Slope_5D": float(latest["VFI_Slope_5D"]) if pd.notna(latest["VFI_Slope_5D"]) else np.nan,
        "VFI_Slope_10D_Normalized": float(latest["VFI_Slope_10D_Normalized"]) if pd.notna(latest["VFI_Slope_10D_Normalized"]) else np.nan,
        "VFI_Slope_20D_Normalized": float(latest["VFI_Slope_20D_Normalized"]) if pd.notna(latest["VFI_Slope_20D_Normalized"]) else np.nan,
        "VFI_Early_Setup": bool(latest["VFI_Early_Setup"]),
        "VFI_Zero_Cross_Up": bool(latest["VFI_Zero_Cross_Up"]),
        "VFI_Zero_Cross_Up_Recent_Near_Zero": bool(latest["VFI_Zero_Cross_Up_Recent_Near_Zero"]),
        "VFI_Estimated_Days_To_Zero": float(latest["VFI_Estimated_Days_To_Zero"]) if pd.notna(latest["VFI_Estimated_Days_To_Zero"]) else np.nan,
        "VFI_Negative_Near_Zero": bool(latest["VFI_Negative_Near_Zero"]),
        "VFI_Positive_Slopes": bool(latest["VFI_Positive_Slopes"]),
        "VFI_Strong_Rising_Slope": bool(latest["VFI_Strong_Rising_Slope"]),
        "Weekly_Bullish_Div_Recent": bool(latest["Weekly_Bullish_Div_Recent"]),
        "Weekly_Bearish_Div_Recent": bool(latest["Weekly_Bearish_Div_Recent"]),
        "Weekly_Div_Signal": str(latest.get("Weekly_Div_Signal", "")),
        "Weekly_Last_Confirmed_Div_Signal": str(latest.get("Weekly_Last_Confirmed_Div_Signal", "")),
        "Weekly_Div_Oscillator": float(latest["Weekly_Div_Value"]) if pd.notna(latest["Weekly_Div_Value"]) else np.nan,
        "Weekly_Div_Osc_Z_52": float(latest["Weekly_Div_Osc_Z_52"]) if pd.notna(latest["Weekly_Div_Osc_Z_52"]) else np.nan,
        "Weekly_Div_Osc_Slope_1W": float(latest["Weekly_Div_Osc_Slope_1W"]) if pd.notna(latest["Weekly_Div_Osc_Slope_1W"]) else np.nan,
        "Weekly_Div_Osc_Slope_2W_Normalized": float(latest["Weekly_Div_Osc_Slope_2W_Normalized"]) if pd.notna(latest["Weekly_Div_Osc_Slope_2W_Normalized"]) else np.nan,
        "Weekly_Div_Osc_Slope_4W_Normalized": float(latest["Weekly_Div_Osc_Slope_4W_Normalized"]) if pd.notna(latest["Weekly_Div_Osc_Slope_4W_Normalized"]) else np.nan,
        "Weekly_Div_Osc_Negative_Near_Zero": bool(latest["Weekly_Div_Osc_Negative_Near_Zero"]),
        "Weekly_Div_Osc_Positive_Slopes": bool(latest["Weekly_Div_Osc_Positive_Slopes"]),
        "Weekly_Div_Osc_Strong_Rising_Slope": bool(latest["Weekly_Div_Osc_Strong_Rising_Slope"]),
        "Weekly_Div_Osc_Early_Setup": bool(latest["Weekly_Div_Osc_Early_Setup"]),
        "Weekly_Div_Osc_Zero_Cross_Up": bool(latest["Weekly_Div_Osc_Zero_Cross_Up"]),
        "Weekly_Div_Osc_Zero_Cross_Up_Recent": bool(latest["Weekly_Div_Osc_Zero_Cross_Up_Recent"]),
        "Weekly_Div_Osc_Estimated_Weeks_To_Zero": float(latest["Weekly_Div_Osc_Estimated_Weeks_To_Zero"]) if pd.notna(latest["Weekly_Div_Osc_Estimated_Weeks_To_Zero"]) else np.nan,
        "Weekly_Div_Age_Days": float(weekly_age) if pd.notna(weekly_age) else np.nan,
        "Weekly_Bars": int(len(weekly)),
        "Weekly_Divergence_History_Sufficient": bool(weekly_history_sufficient),
        "Daily_Div_Signal": str(latest.get("Daily_Div_Signal", "")),
        "Daily_Div_Age_Bars": float(latest["Daily_Div_Age_Bars"]) if pd.notna(latest["Daily_Div_Age_Bars"]) else np.nan,
        "Daily_Bullish_Div_Recent": bool(latest["Daily_Bullish_Div_Recent"]),
        "Signal_Type": signal_type,
        "VFI_Stage": str(latest["VFI_Stage"]),
        "Divergence_Cycle_Stage": str(latest["Divergence_Cycle_Stage"]),
        "VFI_Opportunity_Ready": bool(latest["VFI_Opportunity_Ready"]),
        "Divergence_Cycle_Ready": bool(latest["Divergence_Cycle_Ready"]),
        "Two_Stage_Opportunity": bool(latest["Two_Stage_Opportunity"]),
        "Opportunity_Tier": int(latest["Opportunity_Tier"]),
        "Candidate_Status": str(latest["Candidate_Status"]),
        "Selection_Reason": str(latest["Selection_Reason"]),
        "Setup_Score": float(latest["Setup_Score"]),
        "Bars_History": len(joined),
        "Last_Weekly_Bar": str(weekly.index[-1].date()),
    }
    return events, current


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for strategy, group in events.groupby("Strategy"):
        row = {"Strategy": strategy, "Events": len(group), "Assets": group["Asset"].nunique()}
        for horizon in HORIZONS_DAYS:
            values = pd.to_numeric(group[f"Return_{horizon}D"], errors="coerce").dropna()
            row[f"Valid_{horizon}D"] = int(len(values))
            row[f"Mean_Return_{horizon}D"] = float(values.mean()) if len(values) else np.nan
            row[f"Median_Return_{horizon}D"] = float(values.median()) if len(values) else np.nan
            row[f"Win_Rate_{horizon}D_Pct"] = float((values > 0).mean() * 100) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Win_Rate_20D_Pct", "Mean_Return_20D"], ascending=False, na_position="last")


def main():
    out = Path("reports/vfi_accumulation_divergence")
    out.mkdir(parents=True, exist_ok=True)
    binance = BinanceMarketDataService()
    yahoo = MarketDataService()
    vfi_indicator = VFIIndicator()
    divergence = VolumeDivergenceMM()
    events_all, current_all, errors = [], [], []
    universe_records = discover_crypto_universe(binance, CRYPTO_UNIVERSE_SIZE)
    universe_by_symbol = {row["Symbol"]: row for row in universe_records}
    crypto_symbols = [row["Symbol"] for row in universe_records]
    print(f"Discovered {len(crypto_symbols)} liquid Binance spot/perpetual assets.", flush=True)

    for symbol in crypto_symbols:
        print(f"[BINANCE] {symbol}", flush=True)
        try:
            daily = get_crypto_daily(binance, symbol)
            events, current = build_asset_events(symbol, daily, True, vfi_indicator, divergence)
            current.update(universe_by_symbol[symbol])
            current["Universe_Selection_Method"] = (
                "Top-200 intersection of active USDT spot and USD-M perpetual markets; "
                "ranked by sum of 24h quote-volume ranks, then worst market rank, then symbol"
            )
            events_all.extend(events)
            current_all.append(current)
        except Exception as exc:
            errors.append({"Asset": symbol, "Source": "BINANCE_SPOT", **universe_by_symbol[symbol], "Error": str(exc)})
            print(f"ERROR {symbol}: {exc}", flush=True)

    for asset, ticker in YAHOO_ASSETS.items():
        print(f"[YAHOO] {asset} ({ticker})", flush=True)
        try:
            daily = yahoo.get_daily_data(ticker, period="max")
            events, current = build_asset_events(asset, daily, False, vfi_indicator, divergence)
            current["Yahoo_Ticker"] = ticker
            events_all.extend(events)
            current_all.append(current)
        except Exception as exc:
            errors.append({"Asset": asset, "Source": "YAHOO_FINANCE", "Ticker": ticker, "Error": str(exc)})
            print(f"ERROR {asset} ({ticker}): {exc}", flush=True)

    events = pd.DataFrame(events_all)
    current = pd.DataFrame(current_all)
    errors_df = pd.DataFrame(errors)
    if not current.empty:
        # Ranking is driven by the two-stage cycle, not by a bullish divergence
        # alone or by an oscillator that has already travelled far above zero.
        current["Priority"] = current["Opportunity_Tier"]
        current = current.sort_values(
            ["Priority", "Setup_Score", "VFI_Estimated_Days_To_Zero",
             "Weekly_Div_Osc_Estimated_Weeks_To_Zero"],
            ascending=[True, False, True, True], na_position="last",
        ).reset_index(drop=True)
        current.insert(0, "Rank", np.arange(1, len(current) + 1))
    summary = summarize(events)
    per_asset = (
        events.groupby(["Asset", "Strategy"], dropna=False)
        .agg(Events=("Signal_Date", "count"), Last_Signal=("Signal_Date", "max"))
        .reset_index()
        if not events.empty else pd.DataFrame()
    )

    current.to_csv(out / "current_opportunities.csv", index=False)
    pd.DataFrame(universe_records).to_csv(out / "crypto_universe_liquidity.csv", index=False)
    # Actionable shortlist strictly applies VFI first, then weekly divergence oscillator.
    filtered = current[current["Two_Stage_Opportunity"]].copy() if not current.empty else current.copy()
    filtered = filtered.sort_values(
        ["Opportunity_Tier", "Setup_Score", "VFI_Estimated_Days_To_Zero",
         "Weekly_Div_Osc_Estimated_Weeks_To_Zero"],
        ascending=[True, False, True, True], na_position="last",
    ).reset_index(drop=True)
    filtered.insert(0, "Two_Stage_Rank", np.arange(1, len(filtered) + 1))
    filtered.to_csv(out / "filtered_opportunities.csv", index=False)
    current[current["Candidate_Status"] == "VFI_READY_WAITING_FOR_DIVERGENCE"].to_csv(
        out / "vfi_first_watchlist.csv", index=False
    )
    current[current["Candidate_Status"] == "DIVERGENCE_READY_WAITING_FOR_VFI"].to_csv(
        out / "divergence_watchlist_waiting_for_vfi.csv", index=False
    )
    events.to_csv(out / "historical_events.csv", index=False)
    summary.to_csv(out / "strategy_comparison.csv", index=False)
    per_asset.to_csv(out / "per_asset_strategy_counts.csv", index=False)
    errors_df.to_csv(out / "data_errors.csv", index=False)
    (out / "study_metadata.json").write_text(json.dumps({
        "crypto_assets_requested": len(crypto_symbols),
        "crypto_universe_size": CRYPTO_UNIVERSE_SIZE,
        "crypto_symbols": crypto_symbols,
        "crypto_universe_records": universe_records,
        "universe_selection_method": "Intersection of active Binance USDT spot and USD-M perpetual markets; ranked by sum of 24h quote-volume ranks, then worse market rank, then symbol. Current liquidity snapshot only.",
        "non_crypto_assets_requested": len(YAHOO_ASSETS),
        "yahoo_assets": YAHOO_ASSETS,
        "bar_intervals_used": ["1d", "1w"],
        "intraday_used": False,
        "horizons_trading_days": list(HORIZONS_DAYS),
        "minimum_daily_bars": MIN_DAILY_BARS,
        "minimum_weekly_bars_for_divergence": MIN_WEEKLY_BARS_FOR_DIVERGENCE,
        "weekly_divergence_fresh_days": WEEKLY_DIVERGENCE_FRESH_DAYS,
        "daily_divergence_fresh_days": DAILY_DIVERGENCE_FRESH_DAYS,
        "vfi_early_setup_rule": "VFI <0; VFI_Z_252 in [-0.75, 0); positive 5D/10D/20D slopes; normalized 10D change >=0.20; projected zero cross within 20 trading days.",
        "vfi_fresh_cross_rule": "VFI crossed upward within the last 3 daily bars, VFI_Z_252 <=0.25, and 5D/10D/20D slopes remain positive with a strong normalized 10D change.",
        "weekly_divergence_oscillator_setup_rule": "Weekly Volume Divergence by MM oscillator <0; within 0.75 trailing-52-week standard deviations of zero; 1W/2W/4W slopes positive; normalized 2W change >=0.20; projected zero cross within 8 weeks.",
        "weekly_divergence_fresh_cross_rule": "Weekly oscillator crossed upward within the last 2 completed weekly bars, normalized distance from zero <=0.25, and slopes remain positive/strong.",
        "zero_cross_max_normalized_distance": ZERO_CROSS_MAX_NORMALIZED_DISTANCE,
        "strong_slope_min_normalized_change": STRONG_SLOPE_MIN_NORMALIZED_CHANGE,
        "max_estimated_days_to_vfi_zero": VFI_MAX_ESTIMATED_DAYS_TO_ZERO,
        "max_estimated_weeks_to_divergence_zero": DIVERGENCE_MAX_ESTIMATED_WEEKS_TO_ZERO,
        "divergence": "Causal Volume Divergence by MM implementation; pivot signal recorded on confirmation bar, not backdated. Weekly oscillator cycle is a separate filter from the confirmed price/oscillator bullish-divergence pivot.",
        "completed_assets": int(len(current)),
        "errors": int(len(errors_df)),
        "two_stage_candidates": int(current["Two_Stage_Opportunity"].sum()) if not current.empty else 0,
        "vfi_ready_waiting_for_divergence": int((current["Candidate_Status"] == "VFI_READY_WAITING_FOR_DIVERGENCE").sum()) if not current.empty else 0,
        "divergence_ready_waiting_for_vfi": int((current["Candidate_Status"] == "DIVERGENCE_READY_WAITING_FOR_VFI").sum()) if not current.empty else 0,
    }, indent=2), encoding="utf-8")

    print("\n=== CURRENT OPPORTUNITIES ===")
    if not current.empty:
        cols = [c for c in ["Rank", "Asset", "Universe_Rank", "Date", "Price",
                            "VFI_Daily", "VFI_Z_252", "VFI_Slope_10D_Normalized",
                            "VFI_Estimated_Days_To_Zero", "VFI_Stage",
                            "Weekly_Div_Oscillator", "Weekly_Div_Osc_Z_52",
                            "Weekly_Div_Osc_Slope_2W_Normalized",
                            "Weekly_Div_Osc_Estimated_Weeks_To_Zero",
                            "Divergence_Cycle_Stage", "Two_Stage_Opportunity",
                            "Opportunity_Tier", "Weekly_Bullish_Div_Recent",
                            "Candidate_Status", "Selection_Reason"] if c in current.columns]
        print(current[cols].head(30).to_string(index=False))
        print("\n=== FILTERED TWO-STAGE OPPORTUNITIES ===")
        if not filtered.empty:
            print(filtered[["Two_Stage_Rank", "Asset", "VFI_Daily", "VFI_Estimated_Days_To_Zero",
                            "Weekly_Div_Oscillator", "Weekly_Div_Osc_Estimated_Weeks_To_Zero",
                            "Opportunity_Tier", "Setup_Score", "Selection_Reason"]].head(30).to_string(index=False))
        else:
            print("No assets currently pass both stages. This is a valid result; do not relax filters merely to force picks.")
    print("\n=== STRATEGY COMPARISON ===")
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"\nAssets completed: {len(current)} / {len(crypto_symbols) + len(YAHOO_ASSETS)}")
    print(f"Errors: {len(errors_df)}")
    print(f"Reports: {out.resolve()}")


if __name__ == "__main__":
    main()
