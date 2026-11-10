from __future__ import annotations

"""End-to-end Colombia scanner: weekly VFI, monthly MM divergence, combined rank."""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM
from app.scoring.colombia_opportunity_score import add_colombia_opportunity_score
from app.research.universes import get_universe

OUTPUT_DEFAULT = ROOT / "reports" / "colombia_three_rankings"
RETRIES = 3
RETRY_SLEEP = 3
WEEKLY_EMA_WEIGHTS = {5: 0.05, 20: 0.15, 80: 0.30, 200: 0.50}
MONTHLY_EMA_WEIGHTS = {5: 0.10, 20: 0.30, 80: 0.60}
MONTHLY_VFI_EMA_WEIGHTS = {2: 0.05, 5: 0.15, 18: 0.30, 46: 0.50}
MIN_AVG_MONTHLY_DAILY_VALUE_COP = 300_000_000
MIN_ACTIVE_SESSION_RATIO = 0.70
LIQUIDITY_LOOKBACK_MONTHS = 3


def configure_logging(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "pipeline.log", encoding="utf-8"),
        ],
        force=True,
    )


def download_daily(ticker: str, period: str = "max") -> pd.DataFrame:
    """Retry Yahoo history and fall back to yf.download; validate raw OHLCV."""
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            logging.info("Yahoo download %s attempt %d/%d", ticker, attempt, RETRIES)
            data = yf.Ticker(ticker).history(
                period=period, interval="1d", auto_adjust=True,
                actions=False, repair=False, timeout=30,
            )
            if data is None or data.empty:
                data = yf.download(
                    ticker, period=period, interval="1d",
                    auto_adjust=True, actions=False, progress=False,
                    threads=False, timeout=30,
                )
            if data is None or data.empty:
                raise ValueError("Yahoo returned no rows")
            if isinstance(data.columns, pd.MultiIndex):
                for level in range(data.columns.nlevels):
                    names = set(map(str, data.columns.get_level_values(level)))
                    if {"Open", "High", "Low", "Close", "Volume"}.issubset(names):
                        data.columns = data.columns.get_level_values(level)
                        break
            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required if c not in data.columns]
            if missing:
                raise ValueError(f"Missing OHLCV columns: {missing}")
            data = data[required].copy()
            idx = pd.to_datetime(data.index, errors="coerce")
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            data.index = idx.normalize()
            data = data.loc[~data.index.isna()]
            data = data[~data.index.duplicated(keep="last")].sort_index()
            for col in required:
                data[col] = pd.to_numeric(data[col], errors="coerce")
            data = data.dropna(subset=["Open", "High", "Low", "Close"])
            data["Volume"] = data["Volume"].fillna(0).clip(lower=0)
            if data.empty:
                raise ValueError("No valid OHLCV rows after cleaning")
            if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
                raise ValueError("Non-positive OHLC price")
            # Yahoo's historical Colombian feeds sometimes have High below
            # Open/Close or Low above them. Keep valid extremes and repair only
            # the violated bound using the observed Open/Close values.
            old_high = data["High"].copy()
            old_low = data["Low"].copy()
            data["High"] = data[["High", "Open", "Close", "Low"]].max(axis=1)
            data["Low"] = data[["Low", "Open", "Close", "High"]].min(axis=1)
            repaired = (data["High"] != old_high) | (data["Low"] != old_low)
            if repaired.any():
                logging.warning(
                    "%s: repaired %d inconsistent OHLC sessions (%.2f%%)",
                    ticker, int(repaired.sum()), float(repaired.mean() * 100),
                )
            return data
        except Exception as exc:
            last_error = exc
            logging.warning("%s attempt %d failed: %s", ticker, attempt, exc)
            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP * attempt)
    raise RuntimeError(f"{ticker}: Yahoo download failed after {RETRIES} attempts: {last_error}")


def liquidity_metrics(daily: pd.DataFrame) -> dict:
    """Average daily traded value per completed month, averaged over 3 months.

    Each month's traded value is divided by its observed sessions, then the
    monthly averages are averaged with equal weight. The incomplete month is
    excluded, reducing the impact of isolated high-volume sessions.
    """
    current_month = pd.Timestamp.now().to_period("M")
    eligible = daily.loc[daily.index.to_period("M") < current_month].copy()
    if eligible.empty:
        monthly = pd.DataFrame(columns=["Avg_Daily_Traded_Value_COP", "Active_Session_Ratio", "Sessions"])
    else:
        eligible["Traded_Value_COP"] = (
            eligible["Close"] * eligible["Volume"]
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        eligible["Month"] = eligible.index.to_period("M")
        monthly = eligible.groupby("Month").agg(
            Avg_Daily_Traded_Value_COP=("Traded_Value_COP", "mean"),
            Active_Session_Ratio=("Volume", lambda x: float(x.gt(0).mean())),
            Sessions=("Volume", "size"),
        ).tail(LIQUIDITY_LOOKBACK_MONTHS)

    months = len(monthly)
    avg_monthly_daily_value = float(monthly["Avg_Daily_Traded_Value_COP"].mean()) if months else 0.0
    median_monthly_daily_value = float(monthly["Avg_Daily_Traded_Value_COP"].median()) if months else 0.0
    active_ratio = (
        float(np.average(monthly["Active_Session_Ratio"], weights=monthly["Sessions"]))
        if months else 0.0
    )
    sessions = int(monthly["Sessions"].sum()) if months else 0
    return {
        "Liquidity_Months": int(months),
        "Liquidity_Sessions": sessions,
        "Avg_Monthly_Daily_Traded_Value_COP_3M": avg_monthly_daily_value,
        "Median_Monthly_Daily_Traded_Value_COP_3M": median_monthly_daily_value,
        "Active_Session_Ratio_3M": active_ratio,
        "Liquidity_Pass": bool(
            months >= LIQUIDITY_LOOKBACK_MONTHS
            and avg_monthly_daily_value >= MIN_AVG_MONTHLY_DAILY_VALUE_COP
            and active_ratio >= MIN_ACTIVE_SESSION_RATIO
        ),
    }


def aggregate_bars(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate OHLCV and make empty calendar bars explicit (volume=0)."""
    bars = daily.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    })
    bars = bars.dropna(subset=["Open", "High", "Low", "Close"])
    if bars.empty:
        return bars
    full_idx = pd.date_range(bars.index.min(), bars.index.max(), freq=rule)
    bars = bars.reindex(full_idx)
    synthetic = bars[["Open", "High", "Low", "Close"]].isna().any(axis=1)

    # Carry forward prices only to preserve the calendar index. Never carry
    # forward volume: a synthetic week must contribute zero traded volume.
    for col in ["Open", "High", "Low", "Close"]:
        bars[col] = bars[col].ffill()
    bars["Volume"] = bars["Volume"].fillna(0.0)
    bars["Synthetic_Bar"] = synthetic
    bars.index.name = "Date"
    return bars


# Recent gaps are tolerated only when sparse and not too close to the present.
GAP_LOOKBACK_WEEKS = 52
MAX_RECENT_GAP_RATIO = 0.10
MAX_CONSECUTIVE_RECENT_GAPS = 3
PROTECTED_RECENT_WEEKS = 4


def weekly_gap_metrics(weekly: pd.DataFrame) -> dict:
    """Measure missing/synthetic or zero-volume weeks without fabricating volume."""
    if weekly.empty:
        return {"Recent_Weeks": 0, "Recent_Missing_Weeks": 0,
                "Recent_Missing_Ratio": 1.0, "Max_Consecutive_Recent_Gaps": 0,
                "Last_Gap_Date": "", "Gap_Pass": False}
    synthetic = (weekly["Synthetic_Bar"].fillna(False).astype(bool)
                 if "Synthetic_Bar" in weekly else pd.Series(False, index=weekly.index))
    volume = (pd.to_numeric(weekly["Volume"], errors="coerce").fillna(0)
              if "Volume" in weekly else pd.Series(0, index=weekly.index))
    missing = synthetic | volume.le(0)
    recent = missing.tail(GAP_LOOKBACK_WEEKS)
    # Count consecutive gaps at the end of the series and the maximum run.
    runs = recent.astype(int).groupby((recent != recent.shift()).cumsum()).sum()
    max_run = int(runs.max()) if not runs.empty else 0
    tail_run = 0
    for flag in recent.iloc[::-1]:
        if not flag:
            break
        tail_run += 1
    protected = int(recent.tail(PROTECTED_RECENT_WEEKS).sum())
    ratio = float(recent.mean()) if len(recent) else 1.0
    passed = (len(recent) >= GAP_LOOKBACK_WEEKS
              and ratio <= MAX_RECENT_GAP_RATIO
              and max_run <= MAX_CONSECUTIVE_RECENT_GAPS
              and protected == 0)
    missing_dates = weekly.index[missing]
    return {
        "Recent_Weeks": int(len(recent)),
        "Recent_Missing_Weeks": int(recent.sum()),
        "Recent_Missing_Ratio": ratio,
        "Max_Consecutive_Recent_Gaps": max_run,
        "Consecutive_Gaps_At_End": tail_run,
        "Last_Gap_Date": missing_dates[-1].date().isoformat() if len(missing_dates) else "",
        "Total_Missing_Weeks": int(missing.sum()),
        "Gap_Pass": bool(passed),
    }


def safe_percentile(value: float, history: pd.Series) -> float:
    vals = pd.to_numeric(history, errors="coerce").dropna()
    if not np.isfinite(value) or vals.empty:
        return np.nan
    return float((vals <= value).mean())


def ema_opportunity_score(indicator: pd.Series, weights: dict[int, float],
                          prefix: str, lookback: int = 26) -> tuple[pd.DataFrame, float]:
    """Cross-sectional comparable EMA score; EMA is calculated on indicator values."""
    s = pd.to_numeric(indicator, errors="coerce").astype(float)
    out = pd.DataFrame(index=s.index)
    total = pd.Series(0.0, index=s.index)
    available_weight = pd.Series(0.0, index=s.index)

    for span, weight in weights.items():
        ema = s.ewm(span=span, adjust=False, min_periods=span).mean()
        out[f"{prefix}_EMA_{span}"] = ema
        level_pct = pd.Series(
            [safe_percentile(ema.iloc[i], ema.iloc[:i + 1]) for i in range(len(ema))],
            index=ema.index, dtype=float,
        )
        # Lower/negative historical positioning is preferred for early accumulation.
        level_points = (1.0 - level_pct).clip(0, 1) * 40.0
        level_points = level_points.where(ema <= 0, level_points * (1.0 - level_pct.clip(0, 1)))
        delta = ema.diff()
        positive = delta.gt(0).astype(float)
        persistence = positive.rolling(lookback, min_periods=min(8, lookback)).mean()
        persistence_points = persistence.fillna(0).clip(0, 1) * 35.0
        accel = delta.diff()
        accel_pct = accel.rolling(lookback, min_periods=min(8, lookback)).mean()
        accel_points = (np.tanh(accel_pct.fillna(0) / (ema.abs().rolling(lookback, min_periods=1).median() + 1e-9)) + 1) * 7.5
        # A rising historical average must not hide a currently falling EMA.
        # Penalize every negative latest slope, not only reversals after a long rise.
        negative_slope_penalty = delta.lt(0).astype(float) * 25.0
        prior_rising = persistence.shift(4).ge(0.60)
        returning = prior_rising & delta.le(0)
        return_penalty = returning.astype(float) * 15.0
        positive_penalty = (ema.gt(0).astype(float) * level_pct.fillna(0) * 25.0)
        component = (level_points + persistence_points + accel_points
                     - negative_slope_penalty - return_penalty - positive_penalty).clip(-40, 90)
        out[f"{prefix}_EMA_{span}_Percentile"] = level_pct
        out[f"{prefix}_EMA_{span}_Slope"] = delta
        out[f"{prefix}_EMA_{span}_Persistence"] = persistence
        out[f"{prefix}_EMA_{span}_Negative_Slope_Penalty"] = negative_slope_penalty
        out[f"{prefix}_EMA_{span}_Return_Penalty"] = return_penalty
        out[f"{prefix}_EMA_{span}_Component"] = component
        valid = ema.notna()
        total.loc[valid] += component.loc[valid] * weight
        available_weight.loc[valid] += weight

    score = (total / available_weight.replace(0, np.nan)) * 100.0 / 90.0
    out[f"{prefix}_EMA_Opportunity_Score"] = score
    latest = float(score.dropna().iloc[-1]) if score.notna().any() else np.nan
    return out, latest


def rank_percentile(series: pd.Series) -> pd.Series:
    valid = series.notna()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if valid.any():
        result.loc[valid] = series.loc[valid].rank(pct=True, method="average") * 100.0
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--symbols", default=None)
    args = parser.parse_args()
    output = Path(args.output_dir)
    configure_logging(output)

    symbols = ([x.strip() for x in args.symbols.split(",") if x.strip()]
               if args.symbols else get_universe("COLOMBIA"))
    symbols = list(dict.fromkeys(symbols))
    logging.info("Universe contains %d symbols", len(symbols))

    weekly_rows, monthly_rows, monthly_vfi_rows, metadata, errors = [], [], [], [], []
    liquidity_exclusions = []
    vfi_indicator = VFIIndicator(length=130)
    divergence_indicator = VolumeDivergenceMM(
        vl1=5, vl2=8, lbL=5, lbR=5, range_lower=5, range_upper=60,
        plot_hidden_bull=True, plot_hidden_bear=True,
    )

    for ticker in symbols:
        try:
            daily = download_daily(ticker)
            liq = liquidity_metrics(daily)
            if not liq["Liquidity_Pass"]:
                reason = "Illiquid: average monthly daily traded value COP {:,.0f} (minimum {:,.0f}); active sessions {:.1%} (minimum {:.0%}); months {}".format(
                    liq["Avg_Monthly_Daily_Traded_Value_COP_3M"], MIN_AVG_MONTHLY_DAILY_VALUE_COP,
                    liq["Active_Session_Ratio_3M"], MIN_ACTIVE_SESSION_RATIO,
                    liq["Liquidity_Months"],
                )
                liquidity_exclusions.append({"Symbol": ticker, **liq, "Exclusion_Reason": reason})
                metadata.append({"Symbol": ticker, "Status": "EXCLUDED_ILLIQUID", **liq, "Error": reason})
                logging.info("Excluded %s by liquidity filter: %s", ticker, reason)
                continue
            # Monthly-equivalent VFI is calculated independently of weekly gaps.
            # Use completed monthly OHLCV bars and a true 30-month VFI, never
            # resample or scale the weekly indicator.
            current_month = pd.Timestamp.now().to_period("M")
            completed_months = daily.loc[daily.index.to_period("M") < current_month]
            monthly = aggregate_bars(completed_months, "ME")
            if len(monthly) >= 76:
                monthly_vfi_indicator = VFIIndicator(length=30)
                monthly_vfi = monthly_vfi_indicator.calculate(monthly)
                monthly_vfi_ema, _ = ema_opportunity_score(
                    monthly_vfi["VFI"], MONTHLY_VFI_EMA_WEIGHTS,
                    "Monthly_VFI", lookback=12,
                )
                monthly_vfi = monthly_vfi.join(monthly_vfi_ema)
                mv = monthly_vfi.dropna(subset=["Monthly_VFI_EMA_Opportunity_Score"])
                if not mv.empty:
                    latest_mv = mv.iloc[-1]
                    monthly_vfi_rows.append({
                        "Symbol": ticker,
                        "Latest_Date": mv.index[-1].date().isoformat(),
                        "Latest_VFI_30M": float(latest_mv["VFI"]),
                        "Monthly_VFI_EMA_Opportunity_Score": float(latest_mv["Monthly_VFI_EMA_Opportunity_Score"]),
                        **{f"Monthly_VFI_EMA_{span}": float(latest_mv[f"Monthly_VFI_EMA_{span}"]) if pd.notna(latest_mv[f"Monthly_VFI_EMA_{span}"]) else np.nan for span in MONTHLY_VFI_EMA_WEIGHTS},
                        **{f"Monthly_VFI_EMA_{span}_Slope": float(latest_mv[f"Monthly_VFI_EMA_{span}_Slope"]) if pd.notna(latest_mv[f"Monthly_VFI_EMA_{span}_Slope"]) else np.nan for span in MONTHLY_VFI_EMA_WEIGHTS},
                        "Synthetic_Months": int(monthly["Synthetic_Bar"].sum()),
                        "History_Months": int(len(monthly)), **liq,
                    })
            else:
                logging.info("%s: insufficient monthly VFI history (%d months; need 76)", ticker, len(monthly))

            weekly = aggregate_bars(daily, "W-FRI")
            gap_metrics = weekly_gap_metrics(weekly)
            if not gap_metrics["Gap_Pass"]:
                reason = (
                    "Recent weekly data quality failed: "
                    f"{gap_metrics['Recent_Missing_Weeks']}/{gap_metrics['Recent_Weeks']} "
                    f"missing weeks ({gap_metrics['Recent_Missing_Ratio']:.1%}); "
                    f"max consecutive={gap_metrics['Max_Consecutive_Recent_Gaps']}; "
                    f"last gap={gap_metrics['Last_Gap_Date'] or 'none'}"
                )
                metadata.append({
                    "Symbol": ticker, "Status": "EXCLUDED_WEEKLY_DATA_QUALITY",
                    **gap_metrics, **liq, "Error": reason,
                })
                logging.info("Excluded %s due to recent weekly data quality: %s", ticker, reason)
                continue

            if len(daily) < 180 or len(weekly) < 205 or len(monthly) < 90:
                raise ValueError(
                    f"Insufficient history: daily={len(daily)}, weekly={len(weekly)}, monthly={len(monthly)}"
                )

            vfi = vfi_indicator.calculate(weekly)

            # aggregate_bars already carries the previous complete OHLCV
            # bar into every missing calendar week. Calculate VFI directly
            # on that complete calendar series; do not replace the resulting
            # VFI with NaN/ffill afterwards, because synthetic zero-volume
            # bars would otherwise distort the rolling VFI inputs.
            vfi = add_colombia_opportunity_score(vfi)
            weekly_ema, _ = ema_opportunity_score(vfi["VFI"], WEEKLY_EMA_WEIGHTS, "Weekly_VFI")
            vfi = vfi.join(weekly_ema)
            w = vfi.dropna(subset=["VFI"]).iloc[-1]
            weekly_rows.append({
                "Symbol": ticker,
                "Latest_Date": vfi.dropna(subset=["VFI"]).index[-1].date().isoformat(),
                "Latest_VFI": float(w["VFI"]),
                "VFI_Opportunity_Score": float(w["VFI_Opportunity_Score"]),
                "Weekly_EMA_Opportunity_Score": float(w["Weekly_VFI_EMA_Opportunity_Score"]),
                **{f"Weekly_VFI_EMA_{span}": float(w[f"Weekly_VFI_EMA_{span}"]) if pd.notna(w[f"Weekly_VFI_EMA_{span}"]) else np.nan for span in WEEKLY_EMA_WEIGHTS},
                **{f"Weekly_VFI_EMA_{span}_Slope": float(w[f"Weekly_VFI_EMA_{span}_Slope"]) if pd.notna(w[f"Weekly_VFI_EMA_{span}_Slope"]) else np.nan for span in WEEKLY_EMA_WEIGHTS},
                **{f"Weekly_VFI_EMA_{span}_Component": float(w[f"Weekly_VFI_EMA_{span}_Component"]) if pd.notna(w[f"Weekly_VFI_EMA_{span}_Component"]) else np.nan for span in WEEKLY_EMA_WEIGHTS},
                "VFI_R2_26W": float(w["VFI_R2_26W"]) if pd.notna(w["VFI_R2_26W"]) else np.nan,
                "VFI_State": str(w.get("VFI_State", "")),
                "Synthetic_Weeks": int(weekly["Synthetic_Bar"].sum()), **gap_metrics, **liq,
            })

            div = divergence_indicator.calculate(monthly)
            div_ema, _ = ema_opportunity_score(
                div["Volume_Divergence_MM"], MONTHLY_EMA_WEIGHTS, "Monthly_Divergence"
            )
            div = div.join(div_ema)
            dvalid = div.dropna(subset=["Volume_Divergence_MM"])
            if dvalid.empty or pd.isna(dvalid.iloc[-1]["Monthly_Divergence_EMA_Opportunity_Score"]):
                raise ValueError("Monthly Volume Divergence EMA score is not ready")
            d = dvalid.iloc[-1]
            monthly_rows.append({
                "Symbol": ticker,
                "Latest_Date": dvalid.index[-1].date().isoformat(),
                "Volume_Divergence_MM": float(d["Volume_Divergence_MM"]),
                "Monthly_Divergence_EMA_Opportunity_Score": float(d["Monthly_Divergence_EMA_Opportunity_Score"]),
                "Bullish_Divergence": bool(d["Volume_Div_Bullish"]),
                "Bearish_Divergence": bool(d["Volume_Div_Bearish"]),
                "Bullish_Age_Months": float(d["Volume_Div_Bullish_Age_Bars"]) if pd.notna(d["Volume_Div_Bullish_Age_Bars"]) else np.nan,
                "Bearish_Age_Months": float(d["Volume_Div_Bearish_Age_Bars"]) if pd.notna(d["Volume_Div_Bearish_Age_Bars"]) else np.nan,
                "Synthetic_Months": int(monthly["Synthetic_Bar"].sum()), **liq,
            })
            metadata.append({
                "Symbol": ticker, "Status": "OK", "Daily_Rows": len(daily),
                "Weekly_Rows": len(weekly), "Monthly_Rows": len(monthly), **gap_metrics,
                "First_Date": daily.index.min().date().isoformat(),
                "Last_Date": daily.index.max().date().isoformat(), **liq,
            })
            logging.info("Completed %s", ticker)
        except Exception as exc:
            logging.exception("Failed %s", ticker)
            errors.append({"Symbol": ticker, "Error": f"{type(exc).__name__}: {exc}"})
            metadata.append({"Symbol": ticker, "Status": "ERROR", "Error": str(exc)})
        time.sleep(0.5)

    weekly_df = pd.DataFrame(weekly_rows)
    monthly_df = pd.DataFrame(monthly_rows)
    monthly_vfi_df = pd.DataFrame(monthly_vfi_rows)
    if monthly_vfi_df.empty:
        raise RuntimeError("No valid monthly-equivalent VFI ranking could be produced")
    if monthly_df.empty:
        raise RuntimeError("No valid monthly Volume Divergence ranking could be produced")

    monthly_vfi_df["Rank"] = monthly_vfi_df["Monthly_VFI_EMA_Opportunity_Score"].rank(method="min", ascending=False).astype(int)
    monthly_vfi_df = monthly_vfi_df.sort_values("Monthly_VFI_EMA_Opportunity_Score", ascending=False)

    weekly_df["Rank"] = weekly_df["Weekly_EMA_Opportunity_Score"].rank(
        method="min", ascending=False
    ).astype(int)
    weekly_df = weekly_df.sort_values(["Weekly_EMA_Opportunity_Score"], ascending=False)
    monthly_df["Rank"] = monthly_df["Monthly_Divergence_EMA_Opportunity_Score"].rank(
        method="min", ascending=False
    ).astype(int)
    monthly_df = monthly_df.sort_values(["Monthly_Divergence_EMA_Opportunity_Score"], ascending=False)

    combined = weekly_df[["Symbol", "Weekly_EMA_Opportunity_Score"]].merge(
        monthly_df[["Symbol", "Monthly_Divergence_EMA_Opportunity_Score"]],
        on="Symbol", how="inner", validate="one_to_one",
    )
    combined["Weekly_Percentile"] = rank_percentile(combined["Weekly_EMA_Opportunity_Score"])
    combined["Monthly_Divergence_Percentile"] = rank_percentile(
        combined["Monthly_Divergence_EMA_Opportunity_Score"]
    )
    combined["Combined_Score"] = (
        combined["Weekly_Percentile"] * 0.50
        + combined["Monthly_Divergence_Percentile"] * 0.50
    )
    combined["Rank"] = combined["Combined_Score"].rank(
        method="min", ascending=False
    ).astype(int)
    combined = combined.sort_values("Combined_Score", ascending=False)

    monthly_vfi_df.to_csv(output / "ranking_vfi_mensual_equivalente.csv", index=False)
    weekly_df.to_csv(output / "ranking_vfi_semanal.csv", index=False)
    monthly_df.to_csv(output / "ranking_volume_divergence_mensual.csv", index=False)
    combined.to_csv(output / "ranking_combinado.csv", index=False)
    pd.DataFrame(metadata).to_csv(output / "cobertura_universo.csv", index=False)
    pd.DataFrame(errors, columns=["Symbol", "Error"]).to_csv(output / "errores.csv", index=False)
    pd.DataFrame(liquidity_exclusions).to_csv(output / "excluidos_por_liquidez.csv", index=False)

    run_info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Yahoo Finance via yfinance",
        "symbols_requested": len(symbols),
        "liquidity_filter": {"lookback_completed_months": LIQUIDITY_LOOKBACK_MONTHS, "monthly_metric": "mean of each completed month's daily Close*Volume average, then equal-weight average across months", "minimum_average_monthly_daily_traded_value_cop": MIN_AVG_MONTHLY_DAILY_VALUE_COP, "minimum_active_session_ratio": MIN_ACTIVE_SESSION_RATIO, "excluded_illiquid": len(liquidity_exclusions)},
        "weekly_success": len(weekly_df),
        "monthly_vfi_success": len(monthly_vfi_df),
        "monthly_vfi_length_months": 30,
        "monthly_vfi_ema_spans": MONTHLY_VFI_EMA_WEIGHTS,
        "monthly_vfi_ema_weights": MONTHLY_VFI_EMA_WEIGHTS,
        "monthly_vfi_method": "Recalculated from completed monthly OHLCV bars using VFI length 30; EMA spans 2/5/18/46 on monthly VFI. Independent of weekly gap exclusion.",
        "weekly_gap_policy": {"lookback_weeks": GAP_LOOKBACK_WEEKS, "max_missing_ratio": MAX_RECENT_GAP_RATIO, "max_consecutive_gaps": MAX_CONSECUTIVE_RECENT_GAPS, "protected_latest_weeks": PROTECTED_RECENT_WEEKS, "price_policy": "forward-fill OHLC only for synthetic calendar weeks", "volume_policy": "synthetic calendar weeks retain zero volume; never forward-fill volume"},
        "excluded_weekly_data_quality": sum(1 for row in metadata if row.get("Status") == "EXCLUDED_WEEKLY_DATA_QUALITY"),
        "monthly_success": len(monthly_df),
        "combined_success": len(combined),
        "weekly_ema_weights": WEEKLY_EMA_WEIGHTS,
        "weekly_missing_data_policy": "Allow sparse historical gaps; exclude only if recent 52-week gap ratio exceeds 10%, any run exceeds 3 weeks, fewer than 52 recent weeks exist, or any gap occurs in latest 4 weeks. Synthetic OHLC is carried forward but volume remains zero.",
        "weekly_ema_calculation": "EMA(span) = VFI.ewm(span=span, adjust=False, min_periods=span).mean() on the complete W-FRI calendar-week series",
        "weekly_negative_slope_penalty": "25 points per EMA component whenever its current weekly EMA slope is negative; included in component score",
        "monthly_ema_weights": MONTHLY_EMA_WEIGHTS,
        "combined_weights": {"weekly_percentile": 0.5, "monthly_divergence_percentile": 0.5},
        "weekly_missing_bar_policy": "exclude asset from all rankings only when recent weekly data-quality thresholds fail",
        "monthly_missing_bar_policy": "forward-fill complete OHLCV from the previous calendar month",
        "divergence": "MM WMA cascade 5/8/13/21/34; pivots 5 left/5 right; lookback 5-60; hidden and regular",
        "errors": errors,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logging.info("Liquidity exclusions: %d", len(liquidity_exclusions))
    logging.info("Rankings saved to %s", output)
    if errors:
        logging.warning("%d/%d symbols failed; inspect errores.csv", len(errors), len(symbols))
    print("\nTOP MONTHLY-EQUIVALENT VFI\n", monthly_vfi_df.head(20).to_string(index=False))
    print("\nTOP WEEKLY VFI\n", weekly_df.head(10).to_string(index=False))
    print("\nTOP MONTHLY DIVERGENCE\n", monthly_df.head(10).to_string(index=False))
    print("\nTOP COMBINED\n", combined.head(10).to_string(index=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
