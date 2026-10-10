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

# These are the user's previously discussed research assets, not a newly
# discovered/top-200 universe. Crypto uses Binance spot USDT daily candles.
CRYPTO_SYMBOLS = [
    "AAVEUSDT", "ACEUSDT", "ADAUSDT", "APTUSDT", "ARBUSDT", "ASTERUSDT",
    "AVAXUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT",
    "ENAUSDT", "ETHUSDT", "FETUSDT", "FILUSDT", "GRAMUSDT", "GTCUSDT",
    "HBARUSDT", "HEMIUSDT", "HYPEUSDT", "INJUSDT", "JUPUSDT", "LINKUSDT",
    "LTCUSDT", "MARSCOINUSDT", "METUSDT", "MINAUSDT", "MOVRUSDT", "NEARUSDT",
    "NMRUSDT", "ONDOUSDT", "ORCAUSDT", "PAXGUSDT", "PENGUUSDT", "PUMPUSDT",
    "QNTUSDT", "RENDERUSDT", "SANDUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT",
    "TRUMPUSDT", "TRXUSDT", "UNIUSDT", "WLDUSDT", "XAUTUSDT", "XLMUSDT",
    "XRPUSDT", "ZECUSDT", "ZROUSDT",
]

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
MIN_DAILY_BARS = 260
MIN_WEEKLY_BARS_FOR_DIVERGENCE = 100
WEEKLY_DIVERGENCE_FRESH_DAYS = 35
DAILY_DIVERGENCE_FRESH_DAYS = 20
COOLDOWN_DAILY_BARS = 20


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
        data["VFI_Slope_10D_Normalized"].ge(0.20)
        & data["VFI_Slope_20D_Normalized"].gt(0)
        & data["VFI_Positive_Slopes"]
    )
    data["VFI_Early_Setup"] = (
        data["VFI_Negative_Near_Zero"]
        & data["VFI_Strong_Rising_Slope"]
    )
    data["VFI_Zero_Cross_Up"] = (vfi.shift(1) < 0) & (vfi >= 0) & (s5 > 0)
    data["VFI_Recovery_From_Negative"] = (vfi < 0) & (s5 > 0) & (s10 > 0)
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
    weekly_signal = pd.DataFrame({
        "Weekly_Bullish_Div_Confirmed": wbull,
        "Weekly_Bearish_Div_Confirmed": wbear,
        "Weekly_Bullish_Div_Age_Bars": weekly_bull_age,
        "Weekly_Bearish_Div_Age_Bars": weekly_bear_age,
        "Weekly_Last_Bullish_Signal": weekly_signal_names.where(wbull, np.nan).ffill(),
        "Weekly_Last_Bearish_Signal": weekly_signal_names.where(wbear, np.nan).ffill(),
        "Weekly_Div_Signal": weekly_signal_names,
        "Weekly_Div_Value": weekly_div["Volume_Divergence_MM"],
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

    conditions = {
        "VFI_EARLY_SETUP": joined["VFI_Early_Setup"],
        "VFI_ZERO_CROSS_UP": joined["VFI_Zero_Cross_Up"],
        "WEEKLY_BULLISH_DIVERGENCE": joined["Weekly_Bullish_Div_Recent"],
        "VFI_SETUP_PLUS_WEEKLY_BULL_DIV": joined["VFI_Early_Setup"] & joined["Weekly_Bullish_Div_Recent"],
        "VFI_SETUP_PLUS_DAILY_BULL_DIV": joined["VFI_Early_Setup"] & joined["Daily_Bullish_Div_Recent"],
        "VFI_RECOVERY_PLUS_WEEKLY_BULL_DIV": joined["VFI_Recovery_From_Negative"] & joined["Weekly_Bullish_Div_Recent"],
        "VFI_CROSS_PLUS_WEEKLY_BULL_DIV": joined["VFI_Zero_Cross_Up"] & joined["Weekly_Bullish_Div_Recent"],
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
                "Daily_Div_Signal": str(row.get("Daily_Div_Signal", "")),
                "Daily_Div_Age_Bars": float(row["Daily_Div_Age_Bars"]) if pd.notna(row["Daily_Div_Age_Bars"]) else np.nan,
                "VFI_Early_Setup": bool(row["VFI_Early_Setup"]),
                "VFI_Zero_Cross_Up": bool(row["VFI_Zero_Cross_Up"]),
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
    signal_type = "NO_SETUP"
    if bool(latest["VFI_Early_Setup"]) and bool(latest["Weekly_Bullish_Div_Recent"]):
        signal_type = "VFI_EARLY_SETUP + WEEKLY_BULLISH_DIVERGENCE"
    elif bool(latest["VFI_Early_Setup"]):
        signal_type = "VFI_EARLY_SETUP; WAIT_FOR_DIVERGENCE"
    elif bool(latest["VFI_Recovery_From_Negative"]) and bool(latest["Weekly_Bullish_Div_Recent"]):
        signal_type = "VFI_RECOVERY_FROM_NEGATIVE + WEEKLY_BULLISH_DIVERGENCE"
    elif bool(latest["VFI_Recovery_From_Negative"]):
        signal_type = "VFI_RECOVERY_FROM_NEGATIVE"
    elif bool(latest["VFI_Zero_Cross_Up"]) and bool(latest["Weekly_Bullish_Div_Recent"]):
        signal_type = "VFI_ZERO_CROSS + WEEKLY_BULLISH_DIVERGENCE"
    elif bool(latest["VFI_Zero_Cross_Up"]):
        signal_type = "VFI_ZERO_CROSS_UP"
    elif bool(latest["Weekly_Bullish_Div_Recent"]):
        signal_type = "WEEKLY_BULLISH_DIVERGENCE; WAIT_FOR_VFI_SETUP"
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
        "Weekly_Bullish_Div_Recent": bool(latest["Weekly_Bullish_Div_Recent"]),
        "Weekly_Bearish_Div_Recent": bool(latest["Weekly_Bearish_Div_Recent"]),
        "Weekly_Div_Signal": str(latest.get("Weekly_Div_Signal", "")),
        "Weekly_Last_Confirmed_Div_Signal": str(latest.get("Weekly_Last_Confirmed_Div_Signal", "")),
        "Weekly_Div_Age_Days": float(weekly_age) if pd.notna(weekly_age) else np.nan,
        "Weekly_Bars": int(len(weekly)),
        "Weekly_Divergence_History_Sufficient": bool(weekly_history_sufficient),
        "Daily_Div_Signal": str(latest.get("Daily_Div_Signal", "")),
        "Daily_Div_Age_Bars": float(latest["Daily_Div_Age_Bars"]) if pd.notna(latest["Daily_Div_Age_Bars"]) else np.nan,
        "Daily_Bullish_Div_Recent": bool(latest["Daily_Bullish_Div_Recent"]),
        "Signal_Type": signal_type,
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

    for symbol in CRYPTO_SYMBOLS:
        print(f"[BINANCE] {symbol}", flush=True)
        try:
            daily = get_crypto_daily(binance, symbol)
            events, current = build_asset_events(symbol, daily, True, vfi_indicator, divergence)
            events_all.extend(events)
            current_all.append(current)
        except Exception as exc:
            errors.append({"Asset": symbol, "Source": "BINANCE_SPOT", "Error": str(exc)})
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
        priority = {
            "VFI_EARLY_SETUP + WEEKLY_BULLISH_DIVERGENCE": 1,
            "VFI_ZERO_CROSS + WEEKLY_BULLISH_DIVERGENCE": 2,
            "VFI_EARLY_SETUP; WAIT_FOR_DIVERGENCE": 2,
            "VFI_RECOVERY_FROM_NEGATIVE + WEEKLY_BULLISH_DIVERGENCE": 3,
            "VFI_RECOVERY_FROM_NEGATIVE": 4,
            "VFI_ZERO_CROSS + WEEKLY_BULLISH_DIVERGENCE": 5,
            "VFI_ZERO_CROSS_UP": 6,
            "WEEKLY_BULLISH_DIVERGENCE; WAIT_FOR_VFI_SETUP": 7,
            "NO_SETUP": 8,
        }
        current["Priority"] = current["Signal_Type"].map(priority).fillna(9).astype(int)
        current = current.sort_values(
            ["Priority", "VFI_Slope_10D_Normalized", "VFI_Z_252"],
            ascending=[True, False, False], na_position="last",
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
    events.to_csv(out / "historical_events.csv", index=False)
    summary.to_csv(out / "strategy_comparison.csv", index=False)
    per_asset.to_csv(out / "per_asset_strategy_counts.csv", index=False)
    errors_df.to_csv(out / "data_errors.csv", index=False)
    (out / "study_metadata.json").write_text(json.dumps({
        "crypto_assets_requested": len(CRYPTO_SYMBOLS),
        "non_crypto_assets_requested": len(YAHOO_ASSETS),
        "crypto_symbols": CRYPTO_SYMBOLS,
        "yahoo_assets": YAHOO_ASSETS,
        "bar_intervals_used": ["1d", "1w"],
        "intraday_used": False,
        "horizons_trading_days": list(HORIZONS_DAYS),
        "minimum_daily_bars": MIN_DAILY_BARS,
        "minimum_weekly_bars_for_divergence": MIN_WEEKLY_BARS_FOR_DIVERGENCE,
        "weekly_divergence_fresh_days": WEEKLY_DIVERGENCE_FRESH_DAYS,
        "daily_divergence_fresh_days": DAILY_DIVERGENCE_FRESH_DAYS,
        "vfi_early_setup_rule": "VFI<0; VFI within 0.75 trailing-252-bar standard deviations of zero; 5D/10D/20D slopes positive; 10D change >=0.20 trailing VFI standard deviations.",
        "divergence": "Causal Volume Divergence by MM implementation; pivot signal recorded on confirmation bar, not backdated.",
        "completed_assets": int(len(current)),
        "errors": int(len(errors_df)),
    }, indent=2), encoding="utf-8")

    print("\n=== CURRENT OPPORTUNITIES ===")
    if not current.empty:
        cols = [c for c in ["Rank", "Asset", "Date", "Price", "VFI_Daily", "VFI_Z_252",
                            "VFI_Slope_10D_Normalized", "VFI_Early_Setup",
                            "Weekly_Bullish_Div_Recent", "Weekly_Div_Signal", "Weekly_Div_Age_Days",
                            "Signal_Type"] if c in current.columns]
        print(current[cols].head(30).to_string(index=False))
    print("\n=== STRATEGY COMPARISON ===")
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"\nAssets completed: {len(current)} / {len(CRYPTO_SYMBOLS) + len(YAHOO_ASSETS)}")
    print(f"Errors: {len(errors_df)}")
    print(f"Reports: {out.resolve()}")


if __name__ == "__main__":
    main()
