"""Rank early accumulation candidates using causal, scale-aware comparisons.

The score compares each indicator with its own trailing history, so assets with
different nominal volumes/indicator amplitudes can be compared. Historical
thresholds exclude the current bar; no full-sample normalization is used.
A score is a research ranking, not a probability of profit or a buy instruction.
"""
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM

DEFAULT_SYMBOLS = tuple(
    "DDD,INTC,NVDA,BTC-USD,EC,CIB,MSFT,EEM,SQM,LTC-USD,AU,SPY,AMZN,PBR,NU,QQQ,ISA.CL,GEB.CL,MINEROS.CL,PFAVAL.CL,PFDAVVNDA.CL,TERPEL.CL".split(",")
)
SHORTS = (3, 5, 7, 9)
MEDIUMS = (15, 20, 25, 30)
LONGS = (40, 50, 60, 80)
VFI_LENGTH = 130
# Use the bottom quartile of each asset's own prior absolute indicator values
# as its near-zero zone. These are quantiles, not a shared raw-value threshold.
NEAR_ZERO_QUANTILE = 0.25
WEEKLY_LOOKBACK = 156
DAILY_LOOKBACK = 252
MIN_WEEKLY_BASELINE = 52
MIN_DAILY_BASELINE = 100
OUT = ROOT / "reports" / "early_vfi_accumulation"

# Score weights total 100. Eligibility remains a separate strict gate.
WEIGHTS = {
    "weekly_state": 25, "weekly_ema": 20, "daily_state": 25,
    "daily_ema": 15, "persistence": 15,
}


def ema_combinations():
    return [(s, m, l) for s, m, l in product(SHORTS, MEDIUMS, LONGS) if s < m < l]


def positive_negative_streak(indicator: pd.Series) -> int:
    """Consecutive completed bars ending now with value < 0 and slope > 0."""
    ok = (indicator < 0) & (indicator.diff() > 0)
    count = 0
    for flag in ok.iloc[::-1]:
        if not bool(flag):
            break
        count += 1
    return count


def add_indicator_emas(frame: pd.DataFrame, value_column: str,
                       short: int, medium: int, long: int) -> pd.DataFrame:
    result = frame.copy()
    values = pd.to_numeric(result[value_column], errors="coerce")
    result["EMA_short"] = values.ewm(span=short, adjust=False, min_periods=short).mean()
    result["EMA_medium"] = values.ewm(span=medium, adjust=False, min_periods=medium).mean()
    result["EMA_long"] = values.ewm(span=long, adjust=False, min_periods=long).mean()
    result["Indicator_Slope_1Bar"] = values.diff()
    return result


def _clean_history(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def historical_near_zero_threshold(series: pd.Series, lookback: int,
                                   min_periods: int, quantile: float = NEAR_ZERO_QUANTILE) -> float:
    """Threshold from prior bars only; current observation is deliberately excluded."""
    clean = _clean_history(series)
    if len(clean) < min_periods + 1:
        return float("nan")
    prior = clean.iloc[:-1].tail(lookback).abs()
    if len(prior) < min_periods:
        return float("nan")
    threshold = float(prior.quantile(quantile))
    # An all-zero or near-constant series has no useful scale; don't call it a signal.
    return threshold if np.isfinite(threshold) and threshold > 0 else float("nan")


def historical_percentile(current: float, history: pd.Series,
                          lookback: int, min_periods: int) -> float:
    """Percentile of current value relative to prior observations only (0..1)."""
    # Exclude the last observation in history: it is the current bar whose percentile we measure.
    prior = _clean_history(history.iloc[:-1]).tail(lookback)
    if not np.isfinite(current) or len(prior) < min_periods:
        return 0.5
    return float((prior <= current).mean())


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _latest_scale(series: pd.Series, lookback: int, min_periods: int) -> float:
    prior = _clean_history(series).iloc[:-1].tail(lookback).abs()
    if len(prior) < min_periods:
        return float("nan")
    scale = float(prior.median())
    return scale if np.isfinite(scale) and scale > 0 else float("nan")


def _ema_structure_score(row: pd.Series, scale: float) -> float:
    """Continuous partial credit for the intended negative EMA fan structure."""
    short = float(row["EMA_short"])
    medium = float(row["EMA_medium"])
    long = float(row["EMA_long"])
    if not np.all(np.isfinite([short, medium, long])) or not np.isfinite(scale) or scale <= 0:
        return 0.0
    negative_component = (float(short < 0) + float(medium < 0)) / 2.0
    ordering_component = (float(short > long) + float(medium > long)) / 2.0
    # Reward a visible separation, capped at one robust historical scale.
    separation = max(0.0, min((short + medium) / 2.0 - long, scale)) / scale
    return _clip01(0.4 * negative_component + 0.4 * ordering_component + 0.2 * separation)


def _indicator_state(value: float, slope: float, value_history: pd.Series,
                     slope_history: pd.Series, lookback: int, min_periods: int):
    threshold = historical_near_zero_threshold(value_history, lookback, min_periods)
    near_zero = bool(np.isfinite(threshold) and value < 0 and abs(value) <= threshold)
    negative = bool(np.isfinite(value) and value < 0)
    slope_positive = bool(np.isfinite(slope) and slope > 0)
    slope_percentile = historical_percentile(
        slope, slope_history, lookback, min_periods
    )
    proximity = _clip01(1.0 - abs(value) / threshold) if (
        np.isfinite(threshold) and threshold > 0 and np.isfinite(value) and value < 0
    ) else 0.0
    # Positive slope is useful for this early-accumulation thesis only while
    # the indicator is still negative and in its asset-specific near-zero zone.
    # This prevents already-positive/high indicators from winning on slope alone.
    slope_strength = _clip01((slope_percentile - 0.35) / 0.65)
    near_zero_weight = max(proximity, 0.5 * float(near_zero))
    score = (
        5.0 * float(negative)
        + 10.0 * near_zero_weight
        + 10.0 * slope_strength * float(near_zero)
    )
    return {
        "negative": negative, "near_zero": near_zero,
        "slope_positive": slope_positive, "threshold": threshold,
        "proximity": proximity, "slope_percentile": slope_percentile,
        "score": float(np.clip(score, 0, 25)),
    }


def vfi_positive_extension_penalty(
    current_vfi: float,
    vfi_history: pd.Series,
    lookback: int = WEEKLY_LOOKBACK,
    min_periods: int = MIN_WEEKLY_BASELINE,
) -> dict:
    """Penalize positive VFI according to its percentile in its own raw history.

    Uses the current VFI's empirical percentile against prior raw VFI values,
    not a shared absolute value or absolute-value quantile. Penalty starts at
    historical percentile 75 and reaches 15 points at percentile 95.
    """
    prior = _clean_history(vfi_history).iloc[:-1].tail(lookback)
    result = {
        "VFI_Positive_Extended": False,
        "VFI_Historical_Percentile": float("nan"),
        "VFI_Extension_Threshold": float("nan"),
        "VFI_Extreme_Threshold": float("nan"),
        "VFI_Positive_Extension_Penalty": 0.0,
    }
    if not np.isfinite(current_vfi) or current_vfi <= 0 or len(prior) < min_periods:
        return result

    percentile = float((prior <= current_vfi).mean())
    result["VFI_Historical_Percentile"] = percentile
    result["VFI_Extension_Threshold"] = float(prior.quantile(0.75))
    result["VFI_Extreme_Threshold"] = float(prior.quantile(0.95))
    result["VFI_Positive_Extended"] = bool(percentile > 0.75)
    penalty = 15.0 * float(np.clip((percentile - 0.75) / (0.95 - 0.75), 0.0, 1.0))
    result["VFI_Positive_Extension_Penalty"] = penalty
    return result


def positive_long_ema_penalty(long_ema: float) -> dict:
    """Apply a strong penalty when the long-period weekly VFI EMA is above zero."""
    is_positive = bool(np.isfinite(long_ema) and long_ema > 0)
    return {
        "VFI_Long_EMA_Positive": is_positive,
        "VFI_Long_EMA_Positive_Penalty": 25.0 if is_positive else 0.0,
    }


def _linear_regression_slope_and_r2(series: pd.Series) -> tuple[float, float]:
    """Return linear-regression slope and R² for one indicator series only."""
    values = _clean_history(series).to_numpy(dtype=float)
    if len(values) < 3:
        return float("nan"), float("nan")
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = intercept + slope * x
    total = float(np.sum((values - values.mean()) ** 2))
    residual = float(np.sum((values - fitted) ** 2))
    r2 = 1.0 - residual / total if total > 0 else 0.0
    return float(slope), float(np.clip(r2, 0.0, 1.0))


def descending_slope_penalty(series: pd.Series, scale: float, window: int,
                             max_penalty: float) -> float:
    """Scale-aware penalty for a descending indicator line over the given bars."""
    clean = _clean_history(series)
    values = clean.tail(window)
    if len(values) < window or not np.isfinite(scale) or scale <= 0:
        return 0.0
    # Normalize by typical one-bar movement, not indicator level.
    prior_diffs = clean.diff().iloc[:-1].tail(156).abs().dropna()
    slope_scale = float(prior_diffs.median()) if len(prior_diffs) else float("nan")
    if not np.isfinite(slope_scale) or slope_scale <= 0:
        slope_scale = scale
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, values.to_numpy(dtype=float), 1)[0])
    if not np.isfinite(slope) or slope >= 0:
        return 0.0
    return float(max_penalty * np.clip(-slope / slope_scale, 0.0, 1.0))


def descending_ema_penalty(frame: pd.DataFrame, scale: float,
                           window: int = 4, max_per_ema: float = 5.0) -> dict:
    """Penalize descending short, medium and long EMA curves progressively."""
    result = {
        "EMA_short_Descending_Penalty": descending_slope_penalty(frame["EMA_short"], scale, window, max_per_ema),
        "EMA_medium_Descending_Penalty": descending_slope_penalty(frame["EMA_medium"], scale, window, max_per_ema),
        "EMA_long_Descending_Penalty": descending_slope_penalty(frame["EMA_long"], scale, window, max_per_ema),
    }
    result["Total_Descending_EMA_Penalty"] = float(sum(result.values()))
    return result


def vfi_smoothness_reward(vfi_series: pd.Series, scale: float,
                          window: int = 12, max_reward: float = 10.0) -> dict:
    """Reward a straight, positive-slope trend only on the weekly VFI line.

    R² is calculated exclusively from weekly VFI observations. The reward is
    zero unless the current VFI is negative and the regression slope is positive.
    Stronger positive slopes, smoother paths and negative values near zero score
    more; no divergence or price series is used for R².
    """
    clean = _clean_history(vfi_series)
    recent = clean.tail(window)
    slope, r2 = _linear_regression_slope_and_r2(recent)
    current = float(recent.iloc[-1]) if len(recent) else float("nan")
    result = {
        "VFI_Trend_R2": r2,
        "VFI_Trend_Regression_Slope": slope,
        "VFI_Smoothness_Reward": 0.0,
    }
    if len(recent) < window or not np.isfinite(scale) or scale <= 0:
        return result
    if not np.isfinite(current) or current >= 0 or not np.isfinite(slope) or slope <= 0:
        return result
    slope_strength = float(np.clip(slope / (0.10 * scale), 0.0, 1.0))
    # Proximity to zero boosts the reward, but does not disable it for deeper negative VFI.
    negative_near_zero_factor = float(1.0 / (1.0 + abs(current) / scale))
    positive_step_fraction = float((recent.diff().dropna() > 0).mean())
    reward = max_reward * float(r2) * slope_strength * negative_near_zero_factor * positive_step_fraction
    result["VFI_Smoothness_Reward"] = float(np.clip(reward, 0.0, max_reward))
    return result


def weekly_vfi_trend_penalty(weekly_with_emas: pd.DataFrame) -> dict:
    """Penalize recent bearish EMA crossovers and loss of VFI upslope.

    Each crossover is recent when the latest EMA is below the slower EMA and
    at least one of the prior 3 completed weekly bars was at/above it.
    Penalties are cumulative: short/medium -20, short/long -20, medium/long
    -25. A non-positive latest VFI slope after positive prior slopes costs 7.
    """
    frame = weekly_with_emas.dropna(
        subset=["VFI", "EMA_short", "EMA_medium", "EMA_long", "Indicator_Slope_1Bar"]
    )
    result = {
        "Weekly_EMA_Bearish_Recross": False,
        "Weekly_EMA_Short_Below_Medium_Recross": False,
        "Weekly_EMA_Short_Below_Long_Recross": False,
        "Weekly_EMA_Medium_Below_Long_Recross": False,
        "VFI_Uptrend_Lost": False,
        "Weekly_EMA_Bearish_Recross_Penalty": 0.0,
        "Weekly_EMA_Short_Medium_Cross_Penalty": 0.0,
        "Weekly_EMA_Short_Long_Cross_Penalty": 0.0,
        "Weekly_EMA_Medium_Long_Cross_Penalty": 0.0,
        "VFI_Uptrend_Loss_Penalty": 0.0,
        "Weekly_Trend_Penalty": 0.0,
    }
    if len(frame) < 5:
        return result

    latest = frame.iloc[-1]
    previous_bars = frame.iloc[-4:-1]

    def recent_bearish_cross(fast_name, slow_name):
        return bool(
            latest[fast_name] < latest[slow_name]
            and (previous_bars[fast_name] >= previous_bars[slow_name]).any()
        )

    short_medium = recent_bearish_cross("EMA_short", "EMA_medium")
    short_long = recent_bearish_cross("EMA_short", "EMA_long")
    medium_long = recent_bearish_cross("EMA_medium", "EMA_long")
    prior_slopes = frame["Indicator_Slope_1Bar"].iloc[-4:-1]
    current_slope = float(latest["Indicator_Slope_1Bar"])
    lost_uptrend = bool(current_slope <= 0 and float(prior_slopes.mean()) > 0)

    result["Weekly_EMA_Bearish_Recross"] = short_medium
    result["Weekly_EMA_Short_Below_Medium_Recross"] = short_medium
    result["Weekly_EMA_Short_Below_Long_Recross"] = short_long
    result["Weekly_EMA_Medium_Below_Long_Recross"] = medium_long
    result["VFI_Uptrend_Lost"] = lost_uptrend
    result["Weekly_EMA_Short_Medium_Cross_Penalty"] = 20.0 if short_medium else 0.0
    result["Weekly_EMA_Short_Long_Cross_Penalty"] = 20.0 if short_long else 0.0
    result["Weekly_EMA_Medium_Long_Cross_Penalty"] = 25.0 if medium_long else 0.0
    result["Weekly_EMA_Bearish_Recross_Penalty"] = (
        result["Weekly_EMA_Short_Medium_Cross_Penalty"]
        + result["Weekly_EMA_Short_Long_Cross_Penalty"]
        + result["Weekly_EMA_Medium_Long_Cross_Penalty"]
    )
    result["VFI_Uptrend_Loss_Penalty"] = 7.0 if lost_uptrend else 0.0
    result["Weekly_Trend_Penalty"] = (
        result["Weekly_EMA_Bearish_Recross_Penalty"]
        + result["VFI_Uptrend_Loss_Penalty"]
    )
    return result


def evaluate_combo(symbol, weekly, daily_div, short, medium, long):
    w = add_indicator_emas(weekly, "VFI", short, medium, long)
    d = add_indicator_emas(daily_div, "Volume_Divergence_MM", short, medium, long)
    w_valid = w.dropna(subset=["VFI", "EMA_short", "EMA_medium", "EMA_long", "Indicator_Slope_1Bar"])
    d_valid = d.dropna(subset=["Volume_Divergence_MM", "EMA_short", "EMA_medium", "EMA_long", "Indicator_Slope_1Bar"])
    if w_valid.empty or d_valid.empty:
        raise ValueError(f"{symbol}: insufficient history for EMA combination {short}/{medium}/{long}.")

    wl, dl = w_valid.iloc[-1], d_valid.iloc[-1]
    weekly_value = float(wl["VFI"])
    weekly_slope = float(wl["Indicator_Slope_1Bar"])
    daily_value = float(dl["Volume_Divergence_MM"])
    daily_slope = float(dl["Indicator_Slope_1Bar"])

    w_state = _indicator_state(
        weekly_value, weekly_slope, w["VFI"], w["Indicator_Slope_1Bar"],
        WEEKLY_LOOKBACK, MIN_WEEKLY_BASELINE,
    )
    d_state = _indicator_state(
        daily_value, daily_slope, d["Volume_Divergence_MM"], d["Indicator_Slope_1Bar"],
        DAILY_LOOKBACK, MIN_DAILY_BASELINE,
    )
    weekly_scale = _latest_scale(w["VFI"], WEEKLY_LOOKBACK, MIN_WEEKLY_BASELINE)
    daily_scale = _latest_scale(d["Volume_Divergence_MM"], DAILY_LOOKBACK, MIN_DAILY_BASELINE)

    weekly_ema_score = 20.0 * _ema_structure_score(wl, weekly_scale)
    daily_ema_score = 15.0 * _ema_structure_score(dl, daily_scale)
    weekly_ema_structure = bool(
        wl["EMA_short"] < 0 and wl["EMA_medium"] < 0
        and wl["EMA_short"] > wl["EMA_long"] and wl["EMA_medium"] > wl["EMA_long"]
    )
    daily_ema_structure = bool(
        dl["EMA_short"] < 0 and dl["EMA_medium"] < 0
        and dl["EMA_short"] > dl["EMA_long"] and dl["EMA_medium"] > dl["EMA_long"]
    )
    weekly_pass = bool(weekly_ema_structure and w_state["near_zero"] and w_state["slope_positive"])
    daily_pass = bool(daily_ema_structure and d_state["near_zero"] and d_state["slope_positive"])

    vfi_streak = positive_negative_streak(w["VFI"])
    divergence_streak = positive_negative_streak(d["Volume_Divergence_MM"])
    persistence_score = (
        8.0 * min(vfi_streak / 8.0, 1.0)
        + 7.0 * min(divergence_streak / 10.0, 1.0)
    )
    # Explicitly penalize descending raw VFI slopes and all six EMA curves.
    # Penalties are normalized by each indicator's own trailing scale.
    trend_penalty = weekly_vfi_trend_penalty(w)
    extension_penalty = vfi_positive_extension_penalty(
        weekly_value, w["VFI"], WEEKLY_LOOKBACK, MIN_WEEKLY_BASELINE
    )
    long_ema_penalty = positive_long_ema_penalty(float(wl["EMA_long"]))
    vfi_slope_penalties = {
        "VFI_Slope_1W_Penalty": descending_slope_penalty(w["VFI"], weekly_scale, 2, 5.0),
        "VFI_Slope_4W_Penalty": descending_slope_penalty(w["VFI"], weekly_scale, 4, 5.0),
        "VFI_Slope_8W_Penalty": descending_slope_penalty(w["VFI"], weekly_scale, 8, 5.0),
    }
    vfi_slope_penalties["VFI_Descending_Slope_Penalty"] = float(sum(vfi_slope_penalties.values()))
    weekly_ema_descending = descending_ema_penalty(w, weekly_scale)
    daily_ema_descending = descending_ema_penalty(d, daily_scale)
    smoothness = vfi_smoothness_reward(w["VFI"], weekly_scale, window=12, max_reward=10.0)

    gross_score = (
        w_state["score"] + weekly_ema_score + d_state["score"]
        + daily_ema_score + persistence_score
    )
    total_penalty = (
        trend_penalty["Weekly_Trend_Penalty"]
        + extension_penalty["VFI_Positive_Extension_Penalty"]
        + long_ema_penalty["VFI_Long_EMA_Positive_Penalty"]
        + vfi_slope_penalties["VFI_Descending_Slope_Penalty"]
        + weekly_ema_descending["Total_Descending_EMA_Penalty"]
        + daily_ema_descending["Total_Descending_EMA_Penalty"]
    )
    # Preserve negative scores so descending setups remain visibly penalized.
    score = float(np.clip(gross_score - total_penalty + smoothness["VFI_Smoothness_Reward"], -100, 100))

    return {
        "Symbol": symbol, "EMA_short": short, "EMA_medium": medium, "EMA_long": long,
        "Weekly_Date": str(w.index[-1].date()), "Daily_Date": str(d.index[-1].date()),
        "VFI_130": weekly_value, "VFI_Slope_1W": weekly_slope,
        "VFI_Near_Zero_Threshold": w_state["threshold"],
        "VFI_Slope_Historical_Percentile": w_state["slope_percentile"],
        "EMA_short_VFI": float(wl["EMA_short"]), "EMA_medium_VFI": float(wl["EMA_medium"]),
        "EMA_long_VFI": float(wl["EMA_long"]),
        "Weekly_EMA_Structure_Pass": weekly_ema_structure,
        "Weekly_VFI_Negative": w_state["negative"],
        "Weekly_VFI_Near_Zero": w_state["near_zero"],
        "Weekly_VFI_Positive_Slope": w_state["slope_positive"],
        "Weekly_Filter_Pass": weekly_pass,
        "Volume_Divergence_MM": daily_value, "Volume_Div_Slope_1D": daily_slope,
        "Divergence_Near_Zero_Threshold": d_state["threshold"],
        "Divergence_Slope_Historical_Percentile": d_state["slope_percentile"],
        "EMA_short_Volume_Div": float(dl["EMA_short"]),
        "EMA_medium_Volume_Div": float(dl["EMA_medium"]),
        "EMA_long_Volume_Div": float(dl["EMA_long"]),
        "Daily_EMA_Structure_Pass": daily_ema_structure,
        "Daily_Divergence_Negative": d_state["negative"],
        "Daily_Divergence_Near_Zero": d_state["near_zero"],
        "Daily_Divergence_Positive_Slope": d_state["slope_positive"],
        "Daily_Filter_Pass": daily_pass,
        "VFI_Negative_Positive_Slope_Streak_Weeks": int(vfi_streak),
        "Divergence_Negative_Positive_Slope_Streak_Days": int(divergence_streak),
        "Weekly_State_Score": round(w_state["score"], 4),
        "Weekly_EMA_Score": round(weekly_ema_score, 4),
        "Daily_Divergence_Score": round(d_state["score"], 4),
        "Daily_EMA_Score": round(daily_ema_score, 4),
        "Persistence_Score": round(persistence_score, 4),
        "Gross_Accumulation_Score": round(float(np.clip(gross_score, 0, 100)), 4),
        "Weekly_EMA_Bearish_Recross": trend_penalty["Weekly_EMA_Bearish_Recross"],
        "Weekly_EMA_Short_Below_Medium_Recross": trend_penalty["Weekly_EMA_Short_Below_Medium_Recross"],
        "Weekly_EMA_Short_Below_Long_Recross": trend_penalty["Weekly_EMA_Short_Below_Long_Recross"],
        "Weekly_EMA_Medium_Below_Long_Recross": trend_penalty["Weekly_EMA_Medium_Below_Long_Recross"],
        "VFI_Uptrend_Lost": trend_penalty["VFI_Uptrend_Lost"],
        "Weekly_EMA_Bearish_Recross_Penalty": trend_penalty["Weekly_EMA_Bearish_Recross_Penalty"],
        "Weekly_EMA_Short_Medium_Cross_Penalty": trend_penalty["Weekly_EMA_Short_Medium_Cross_Penalty"],
        "Weekly_EMA_Short_Long_Cross_Penalty": trend_penalty["Weekly_EMA_Short_Long_Cross_Penalty"],
        "Weekly_EMA_Medium_Long_Cross_Penalty": trend_penalty["Weekly_EMA_Medium_Long_Cross_Penalty"],
        "VFI_Uptrend_Loss_Penalty": trend_penalty["VFI_Uptrend_Loss_Penalty"],
        "Weekly_Trend_Penalty": trend_penalty["Weekly_Trend_Penalty"],
        "VFI_Positive_Extended": extension_penalty["VFI_Positive_Extended"],
        "VFI_Historical_Percentile": round(extension_penalty["VFI_Historical_Percentile"], 4) if np.isfinite(extension_penalty["VFI_Historical_Percentile"]) else float("nan"),
        "VFI_Extension_Threshold": extension_penalty["VFI_Extension_Threshold"],
        "VFI_Extreme_Threshold": extension_penalty["VFI_Extreme_Threshold"],
        "VFI_Positive_Extension_Penalty": round(extension_penalty["VFI_Positive_Extension_Penalty"], 4),
        "VFI_Long_EMA_Positive": long_ema_penalty["VFI_Long_EMA_Positive"],
        "VFI_Long_EMA_Positive_Penalty": long_ema_penalty["VFI_Long_EMA_Positive_Penalty"],
        **vfi_slope_penalties,
        "VFI_EMA_short_Descending_Penalty": round(weekly_ema_descending["EMA_short_Descending_Penalty"], 4),
        "VFI_EMA_medium_Descending_Penalty": round(weekly_ema_descending["EMA_medium_Descending_Penalty"], 4),
        "VFI_EMA_long_Descending_Penalty": round(weekly_ema_descending["EMA_long_Descending_Penalty"], 4),
        "VFI_Total_Descending_EMA_Penalty": round(weekly_ema_descending["Total_Descending_EMA_Penalty"], 4),
        "Divergence_EMA_short_Descending_Penalty": round(daily_ema_descending["EMA_short_Descending_Penalty"], 4),
        "Divergence_EMA_medium_Descending_Penalty": round(daily_ema_descending["EMA_medium_Descending_Penalty"], 4),
        "Divergence_EMA_long_Descending_Penalty": round(daily_ema_descending["EMA_long_Descending_Penalty"], 4),
        "Divergence_Total_Descending_EMA_Penalty": round(daily_ema_descending["Total_Descending_EMA_Penalty"], 4),
        "VFI_Trend_R2": round(smoothness["VFI_Trend_R2"], 6) if np.isfinite(smoothness["VFI_Trend_R2"]) else float("nan"),
        "VFI_Trend_Regression_Slope": smoothness["VFI_Trend_Regression_Slope"],
        "VFI_Smoothness_Reward": round(smoothness["VFI_Smoothness_Reward"], 4),
        "Total_Score_Penalty": round(total_penalty, 4),
        "Accumulation_Score": round(score, 4),
        "Final_Filter_Pass": bool(weekly_pass and daily_pass),
    }


def build_asset_summary(all_combos: pd.DataFrame) -> pd.DataFrame:
    """Choose the strongest combo per asset; eligibility is never inferred from score."""
    summaries = []
    for symbol, group in all_combos.groupby("Symbol", sort=False):
        # Do not gate or prioritize assets by a binary pass/fail flag.
        # Choose the highest-scoring EMA combination; retain pass flags as diagnostics.
        ranked = group.sort_values(
            ["Accumulation_Score",
             "VFI_Negative_Positive_Slope_Streak_Weeks",
             "Divergence_Negative_Positive_Slope_Streak_Days"],
            ascending=[False, False, False],
        )
        best = ranked.iloc[0]
        summaries.append({
            "Symbol": symbol,
            "Accumulation_Score": float(best["Accumulation_Score"]),
            "Weekly_State_Score": float(best["Weekly_State_Score"]),
            "Weekly_EMA_Score": float(best["Weekly_EMA_Score"]),
            "Daily_Divergence_Score": float(best["Daily_Divergence_Score"]),
            "Daily_EMA_Score": float(best["Daily_EMA_Score"]),
            "Persistence_Score": float(best["Persistence_Score"]),
            "Gross_Accumulation_Score": float(best["Gross_Accumulation_Score"]),
            "Weekly_EMA_Bearish_Recross": bool(best["Weekly_EMA_Bearish_Recross"]),
            "Weekly_EMA_Short_Below_Medium_Recross": bool(best["Weekly_EMA_Short_Below_Medium_Recross"]),
            "Weekly_EMA_Short_Below_Long_Recross": bool(best["Weekly_EMA_Short_Below_Long_Recross"]),
            "Weekly_EMA_Medium_Below_Long_Recross": bool(best["Weekly_EMA_Medium_Below_Long_Recross"]),
            "VFI_Uptrend_Lost": bool(best["VFI_Uptrend_Lost"]),
            "Weekly_EMA_Bearish_Recross_Penalty": float(best["Weekly_EMA_Bearish_Recross_Penalty"]),
            "Weekly_EMA_Short_Medium_Cross_Penalty": float(best["Weekly_EMA_Short_Medium_Cross_Penalty"]),
            "Weekly_EMA_Short_Long_Cross_Penalty": float(best["Weekly_EMA_Short_Long_Cross_Penalty"]),
            "Weekly_EMA_Medium_Long_Cross_Penalty": float(best["Weekly_EMA_Medium_Long_Cross_Penalty"]),
            "VFI_Uptrend_Loss_Penalty": float(best["VFI_Uptrend_Loss_Penalty"]),
            "Weekly_Trend_Penalty": float(best["Weekly_Trend_Penalty"]),
            "VFI_Positive_Extended": bool(best["VFI_Positive_Extended"]),
            "VFI_Historical_Percentile": float(best["VFI_Historical_Percentile"]),
            "VFI_Extension_Threshold": best["VFI_Extension_Threshold"],
            "VFI_Extreme_Threshold": best["VFI_Extreme_Threshold"],
            "VFI_Positive_Extension_Penalty": float(best["VFI_Positive_Extension_Penalty"]),
            "VFI_Long_EMA_Positive": bool(best["VFI_Long_EMA_Positive"]),
            "VFI_Long_EMA_Positive_Penalty": float(best["VFI_Long_EMA_Positive_Penalty"]),
            "VFI_Slope_1W_Penalty": float(best["VFI_Slope_1W_Penalty"]),
            "VFI_Slope_4W_Penalty": float(best["VFI_Slope_4W_Penalty"]),
            "VFI_Slope_8W_Penalty": float(best["VFI_Slope_8W_Penalty"]),
            "VFI_Descending_Slope_Penalty": float(best["VFI_Descending_Slope_Penalty"]),
            "VFI_EMA_short_Descending_Penalty": float(best["VFI_EMA_short_Descending_Penalty"]),
            "VFI_EMA_medium_Descending_Penalty": float(best["VFI_EMA_medium_Descending_Penalty"]),
            "VFI_EMA_long_Descending_Penalty": float(best["VFI_EMA_long_Descending_Penalty"]),
            "VFI_Total_Descending_EMA_Penalty": float(best["VFI_Total_Descending_EMA_Penalty"]),
            "Divergence_EMA_short_Descending_Penalty": float(best["Divergence_EMA_short_Descending_Penalty"]),
            "Divergence_EMA_medium_Descending_Penalty": float(best["Divergence_EMA_medium_Descending_Penalty"]),
            "Divergence_EMA_long_Descending_Penalty": float(best["Divergence_EMA_long_Descending_Penalty"]),
            "Divergence_Total_Descending_EMA_Penalty": float(best["Divergence_Total_Descending_EMA_Penalty"]),
            "VFI_Trend_R2": float(best["VFI_Trend_R2"]),
            "VFI_Trend_Regression_Slope": float(best["VFI_Trend_Regression_Slope"]),
            "VFI_Smoothness_Reward": float(best["VFI_Smoothness_Reward"]),
            "Total_Score_Penalty": float(best["Total_Score_Penalty"]),
            "Weekly_Combos_Passing": int(group["Weekly_Filter_Pass"].sum()),
            "Daily_Combos_Passing": int(group["Daily_Filter_Pass"].sum()),
            "Final_Combos_Passing": int(group["Final_Filter_Pass"].sum()),
            "Weekly_Filter_Pass": bool(best["Weekly_Filter_Pass"]),
            "Daily_Filter_Pass": bool(best["Daily_Filter_Pass"]),
            "Eligible": bool(best["Final_Filter_Pass"]),
            "Diagnostic_Only": bool(not best["Final_Filter_Pass"]),
            "Best_EMA_short": int(best["EMA_short"]),
            "Best_EMA_medium": int(best["EMA_medium"]),
            "Best_EMA_long": int(best["EMA_long"]),
            "VFI_130": float(best["VFI_130"]),
            "VFI_Slope_1W": float(best["VFI_Slope_1W"]),
            "VFI_Slope_Historical_Percentile": float(best["VFI_Slope_Historical_Percentile"]),
            "VFI_Near_Zero_Threshold": best["VFI_Near_Zero_Threshold"],
            "Weekly_VFI_Near_Zero": bool(best["Weekly_VFI_Near_Zero"]),
            "Weekly_VFI_Positive_Slope": bool(best["Weekly_VFI_Positive_Slope"]),
            "Weekly_EMA_Structure_Pass": bool(best["Weekly_EMA_Structure_Pass"]),
            "Volume_Divergence_MM": float(best["Volume_Divergence_MM"]),
            "Volume_Div_Slope_1D": float(best["Volume_Div_Slope_1D"]),
            "Divergence_Slope_Historical_Percentile": float(best["Divergence_Slope_Historical_Percentile"]),
            "Divergence_Near_Zero_Threshold": best["Divergence_Near_Zero_Threshold"],
            "Daily_Divergence_Near_Zero": bool(best["Daily_Divergence_Near_Zero"]),
            "Daily_Divergence_Positive_Slope": bool(best["Daily_Divergence_Positive_Slope"]),
            "Daily_EMA_Structure_Pass": bool(best["Daily_EMA_Structure_Pass"]),
            "VFI_Negative_Positive_Slope_Streak_Weeks": int(best["VFI_Negative_Positive_Slope_Streak_Weeks"]),
            "Divergence_Negative_Positive_Slope_Streak_Days": int(best["Divergence_Negative_Positive_Slope_Streak_Days"]),
            "Weekly_Date": best["Weekly_Date"], "Daily_Date": best["Daily_Date"],
        })
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary["Rank"] = summary["Accumulation_Score"].rank(method="min", ascending=False).astype(int)
        summary["Rank_Percentile"] = (
            summary["Accumulation_Score"].rank(method="average", pct=True) * 100
        ).round(2)
        summary = summary.sort_values(
            ["Accumulation_Score", "Rank"],
            ascending=[False, True],
        ).reset_index(drop=True)
    return summary


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args(argv)
    symbols = list(dict.fromkeys(s.strip() for s in args.symbols.split(",") if s.strip()))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    market = MarketDataService()
    rows, errors = [], []

    for symbol in symbols:
        print("=" * 72)
        print(f"EARLY ACCUMULATION SCAN: {symbol}", flush=True)
        try:
            daily = market.get_daily_data(symbol, period="max").sort_index()
            if daily.empty:
                raise ValueError("Market data provider returned no rows.")
            # Do not use today's potentially incomplete daily candle.
            daily = daily[daily.index.normalize() < pd.Timestamp.now().normalize()]
            if daily.empty:
                raise ValueError("No completed daily candles remain.")
            weekly_ohlcv = market.convert_to_weekly(daily).sort_index()
            weekly = VFIIndicator(length=VFI_LENGTH).calculate(weekly_ohlcv)
            weekly = weekly.replace([np.inf, -np.inf], np.nan).dropna(subset=["VFI"])
            daily_div = VolumeDivergenceMM().calculate(daily)
            daily_div = daily_div.replace([np.inf, -np.inf], np.nan).dropna(subset=["Volume_Divergence_MM"])
            if weekly.empty or daily_div.empty:
                raise ValueError("No valid data after indicator warm-up.")
            # A stable baseline is required for comparable scores. Do not silently
            # assign top scores to assets with too little history.
            minimum_valid_weekly = MIN_WEEKLY_BASELINE + max(LONGS) + 1
            if len(weekly) < minimum_valid_weekly:
                raise ValueError(
                    f"Insufficient valid weekly VFI history for normalized score: {len(weekly)} bars; "
                    f"need at least {minimum_valid_weekly} after VFI warm-up."
                )
            for short, medium, long in ema_combinations():
                rows.append(evaluate_combo(symbol, weekly, daily_div, short, medium, long))
            print(
                f"Weekly bars: {len(weekly)}; daily divergence bars: {len(daily_div)}; "
                f"data through {daily_div.index[-1].date()}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc)})
            print(f"SKIPPED {symbol}: {exc}", flush=True)

    if rows:
        all_combos = pd.DataFrame(rows)
        all_combos.to_csv(output / "ema_combo_scan.csv", index=False)
        summary = build_asset_summary(all_combos)
        summary.to_csv(output / "early_vfi_accumulation_ranking.csv", index=False)
        # Separate file makes it harder to mistake diagnostic candidates for signals.
        summary[summary["Eligible"]].to_csv(output / "qualified_accumulation_candidates.csv", index=False)
        summary[~summary["Eligible"]].to_csv(output / "diagnostic_candidates_not_qualified.csv", index=False)
    else:
        pd.DataFrame().to_csv(output / "ema_combo_scan.csv", index=False)
        pd.DataFrame().to_csv(output / "early_vfi_accumulation_ranking.csv", index=False)
        pd.DataFrame().to_csv(output / "qualified_accumulation_candidates.csv", index=False)
        pd.DataFrame().to_csv(output / "diagnostic_candidates_not_qualified.csv", index=False)
    pd.DataFrame(errors, columns=["Symbol", "Error"]).to_csv(
        output / "early_vfi_accumulation_errors.csv", index=False
    )
    print(f"Reports written to: {output}")
    qualified_count = int(summary["Eligible"].sum()) if rows else 0
    print(f"Assets with diagnostic weekly + daily pass flags: {qualified_count}")
    print(f"Assets ranked: {len(set(r['Symbol'] for r in rows))}; EMA combinations per asset: {len(ema_combinations())}")
    print("All assets are ranked by score; pass/fail columns are diagnostics only, not filters.")
    print("Score is comparative, not a probability of profit.")


if __name__ == "__main__":
    main()
