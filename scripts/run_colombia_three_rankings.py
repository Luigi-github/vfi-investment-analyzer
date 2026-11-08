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
    synthetic = bars["Close"].isna()
    for col in ["Open", "High", "Low", "Close"]:
        bars[col] = bars[col].ffill()
    bars.loc[synthetic, "Volume"] = 0.0
    bars["Volume"] = bars["Volume"].fillna(0.0)
    bars["Synthetic_Bar"] = synthetic
    bars.index.name = "Date"
    return bars


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
        # Explicitly penalize a deteriorating EMA after a prior rising run.
        prior_rising = persistence.shift(4).ge(0.60)
        returning = prior_rising & delta.le(0)
        return_penalty = returning.astype(float) * 15.0
        # Positive EMA levels incur an increasingly severe extension penalty.
        positive_penalty = (ema.gt(0).astype(float) * level_pct.fillna(0) * 25.0)
        component = (level_points + persistence_points + accel_points
                     - return_penalty - positive_penalty).clip(-40, 90)
        out[f"{prefix}_EMA_{span}_Percentile"] = level_pct
        out[f"{prefix}_EMA_{span}_Persistence"] = persistence
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

    weekly_rows, monthly_rows, metadata, errors = [], [], [], []
    vfi_indicator = VFIIndicator(length=130)
    divergence_indicator = VolumeDivergenceMM(
        vl1=5, vl2=8, lbL=5, lbR=5, range_lower=5, range_upper=60,
        plot_hidden_bull=True, plot_hidden_bear=True,
    )

    for ticker in symbols:
        try:
            daily = download_daily(ticker)
            weekly = aggregate_bars(daily, "W-FRI")
            # Never rank an unfinished calendar month: Yahoo labels a partial
            # November bar as November 30, which would falsely look future-dated.
            current_month = pd.Timestamp.now().to_period("M")
            completed_months = daily.loc[daily.index.to_period("M") < current_month]
            monthly = aggregate_bars(completed_months, "ME")
            if len(daily) < 180 or len(weekly) < 205 or len(monthly) < 90:
                raise ValueError(
                    f"Insufficient history: daily={len(daily)}, weekly={len(weekly)}, monthly={len(monthly)}"
                )

            vfi = vfi_indicator.calculate(weekly)
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
                "VFI_R2_26W": float(w["VFI_R2_26W"]) if pd.notna(w["VFI_R2_26W"]) else np.nan,
                "VFI_State": str(w.get("VFI_State", "")),
                "Synthetic_Weeks": int(weekly["Synthetic_Bar"].sum()),
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
                "Synthetic_Months": int(monthly["Synthetic_Bar"].sum()),
            })
            metadata.append({
                "Symbol": ticker, "Status": "OK", "Daily_Rows": len(daily),
                "Weekly_Rows": len(weekly), "Monthly_Rows": len(monthly),
                "First_Date": daily.index.min().date().isoformat(),
                "Last_Date": daily.index.max().date().isoformat(),
            })
            logging.info("Completed %s", ticker)
        except Exception as exc:
            logging.exception("Failed %s", ticker)
            errors.append({"Symbol": ticker, "Error": f"{type(exc).__name__}: {exc}"})
            metadata.append({"Symbol": ticker, "Status": "ERROR", "Error": str(exc)})
        time.sleep(0.5)

    weekly_df = pd.DataFrame(weekly_rows)
    monthly_df = pd.DataFrame(monthly_rows)
    if weekly_df.empty:
        raise RuntimeError("No valid weekly VFI ranking could be produced")
    if monthly_df.empty:
        raise RuntimeError("No valid monthly Volume Divergence ranking could be produced")

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

    weekly_df.to_csv(output / "ranking_vfi_semanal.csv", index=False)
    monthly_df.to_csv(output / "ranking_volume_divergence_mensual.csv", index=False)
    combined.to_csv(output / "ranking_combinado.csv", index=False)
    pd.DataFrame(metadata).to_csv(output / "cobertura_universo.csv", index=False)
    pd.DataFrame(errors, columns=["Symbol", "Error"]).to_csv(output / "errores.csv", index=False)

    run_info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Yahoo Finance via yfinance",
        "symbols_requested": len(symbols),
        "weekly_success": len(weekly_df),
        "monthly_success": len(monthly_df),
        "combined_success": len(combined),
        "weekly_ema_weights": WEEKLY_EMA_WEIGHTS,
        "monthly_ema_weights": MONTHLY_EMA_WEIGHTS,
        "combined_weights": {"weekly_percentile": 0.5, "monthly_divergence_percentile": 0.5},
        "weekly_missing_bar_policy": "forward-fill OHLC, set synthetic volume to zero",
        "monthly_missing_bar_policy": "forward-fill OHLC, set synthetic volume to zero",
        "divergence": "MM WMA cascade 5/8/13/21/34; pivots 5 left/5 right; lookback 5-60; hidden and regular",
        "errors": errors,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logging.info("Rankings saved to %s", output)
    if errors:
        logging.warning("%d/%d symbols failed; inspect errores.csv", len(errors), len(symbols))
    print("\nTOP WEEKLY VFI\n", weekly_df.head(10).to_string(index=False))
    print("\nTOP MONTHLY DIVERGENCE\n", monthly_df.head(10).to_string(index=False))
    print("\nTOP COMBINED\n", combined.head(10).to_string(index=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
