"""Walk-forward parameter tuning for the example-asset early-accumulation theory.

Uses only completed daily candles for VFI and completed weekly candles for the
TradingView-style Volume Divergence by MM oscillator. Intraday data is excluded.
Parameters are selected on each asset's first 70% of history and evaluated on
the last 30%; the holdout results are not used to select the winner.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM

ASSETS = {
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
VFI_FORMULA_GRID = {
    "length": [104, 130, 156],
    "coef": [0.15, 0.20, 0.25],
    "vcoef": [2.0, 2.5, 3.0],
}
VFI_THRESHOLD_GRID = {
    "vfi_near_zero_z": [0.50, 0.75, 1.00],
    "vfi_slope_min": [0.10, 0.20, 0.30],
    "vfi_max_days_to_zero": [10.0, 20.0, 30.0],
}
DIV_THRESHOLD_GRID = {
    "div_near_zero_z": [0.50, 0.75, 1.00],
    "div_slope_min": [0.10, 0.20, 0.30],
    "div_max_weeks_to_zero": [4.0, 8.0, 12.0],
}
HORIZONS = (20, 40, 60)  # trading days; deliberately no intraday or 1-10 day targets
ROUND_TRIP_COST_PCT = 0.25
MIN_DAILY_BARS = 260
MIN_WEEKLY_BARS = 129  # 77-bar WMA warm-up + 52 valid weekly oscillator bars
COOLDOWN_DAILY_BARS = 20


def completed_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.sort_index().copy()
    idx = pd.DatetimeIndex(data.index)
    today = pd.Timestamp.now(tz=idx.tz).normalize() if idx.tz is not None else pd.Timestamp.now().normalize()
    normalized = idx.normalize()
    return data.loc[normalized < today].loc[lambda x: ~x.index.duplicated(keep="last")].dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]
    )


def completed_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    weekly = daily.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    last = pd.Timestamp(daily.index[-1]).normalize()
    return weekly.loc[pd.DatetimeIndex(weekly.index).normalize() <= last]


def vfi_features(daily: pd.DataFrame, formula: dict) -> pd.DataFrame:
    frame = VFIIndicator(
        length=int(formula["length"]),
        coef=float(formula["coef"]),
        vcoef=float(formula["vcoef"]),
        signal_length=20,
        signal_length_50=50,
        smooth_vfi=False,
    ).calculate(daily)
    vfi = frame["VFI"].astype(float)
    scale = vfi.rolling(252, min_periods=60).std().replace(0, np.nan)
    frame["VFI_Z"] = vfi / scale
    for n in (5, 10, 20):
        frame[f"SLOPE_{n}"] = (vfi - vfi.shift(n)) / n
        frame[f"SLOPE_{n}_Z"] = (vfi - vfi.shift(n)) / scale
    frame["DAYS_TO_ZERO"] = np.where((vfi < 0) & (frame["SLOPE_10"] > 0), -vfi / frame["SLOPE_10"], np.nan)
    return frame


def divergence_features(weekly: pd.DataFrame) -> pd.DataFrame:
    # Preserve the TradingView Volume Divergence by MM default oscillator
    # construction (5/8/13/21/34 WMA chain, 5-left/5-right pivots).
    frame = VolumeDivergenceMM(vl1=5, vl2=8, lbL=5, lbR=5, range_upper=60, range_lower=5).calculate(weekly)
    osc = frame["Volume_Divergence_MM"].astype(float)
    scale = osc.rolling(52, min_periods=20).std().replace(0, np.nan)
    frame["DIV_Z"] = osc / scale
    for n in (1, 2, 4):
        frame[f"DIV_SLOPE_{n}"] = (osc - osc.shift(n)) / n
        frame[f"DIV_SLOPE_{n}_Z"] = (osc - osc.shift(n)) / scale
    frame["WEEKS_TO_ZERO"] = np.where((osc < 0) & (frame["DIV_SLOPE_2"] > 0), -osc / frame["DIV_SLOPE_2"], np.nan)
    return frame


def vfi_setup(frame: pd.DataFrame, p: dict) -> pd.Series:
    return (
        frame["VFI"].lt(0)
        & frame["VFI_Z"].ge(-float(p["vfi_near_zero_z"]))
        & frame["SLOPE_5"].gt(0)
        & frame["SLOPE_10"].gt(0)
        & frame["SLOPE_20"].gt(0)
        & frame["SLOPE_10_Z"].ge(float(p["vfi_slope_min"]))
        & frame["SLOPE_20_Z"].gt(0)
        & frame["DAYS_TO_ZERO"].between(0, float(p["vfi_max_days_to_zero"]))
    ).fillna(False)


def divergence_setup(frame: pd.DataFrame, p: dict) -> pd.Series:
    return (
        frame["Volume_Divergence_MM"].lt(0)
        & frame["DIV_Z"].ge(-float(p["div_near_zero_z"]))
        & frame["DIV_SLOPE_1"].gt(0)
        & frame["DIV_SLOPE_2"].gt(0)
        & frame["DIV_SLOPE_4"].gt(0)
        & frame["DIV_SLOPE_2_Z"].ge(float(p["div_slope_min"]))
        & frame["DIV_SLOPE_4_Z"].gt(0)
        & frame["WEEKS_TO_ZERO"].between(0, float(p["div_max_weeks_to_zero"]))
    ).fillna(False)


def event_positions(mask: pd.Series, cooldown: int = COOLDOWN_DAILY_BARS) -> list[int]:
    values = mask.fillna(False).to_numpy(dtype=bool)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    kept, last = [], -10**9
    for pos in starts:
        if int(pos) - last >= cooldown:
            kept.append(int(pos))
            last = int(pos)
    return kept


def summarize_events(events: list[dict], prefix: str) -> dict:
    out = {}
    for h in HORIZONS:
        vals = [e[f"R{h}"] for e in events if pd.notna(e.get(f"R{h}"))]
        if vals:
            vals = np.asarray(vals, dtype=float)
            out[f"{prefix}_Events_{h}D"] = int(len(vals))
            out[f"{prefix}_Mean_Net_Return_{h}D_Pct"] = float(np.mean(vals))
            out[f"{prefix}_Median_Net_Return_{h}D_Pct"] = float(np.median(vals))
            out[f"{prefix}_Win_Rate_{h}D_Pct"] = float(np.mean(vals > 0) * 100)
        else:
            out[f"{prefix}_Events_{h}D"] = 0
            out[f"{prefix}_Mean_Net_Return_{h}D_Pct"] = np.nan
            out[f"{prefix}_Median_Net_Return_{h}D_Pct"] = np.nan
            out[f"{prefix}_Win_Rate_{h}D_Pct"] = np.nan
    if events:
        per_asset = {}
        for e in events:
            per_asset.setdefault(e["Asset"], []).append(e["R40"])
        asset_means = [np.mean([v for v in vals if pd.notna(v)]) for vals in per_asset.values() if any(pd.notna(v) for v in vals)]
        out[f"{prefix}_Assets_40D"] = len(asset_means)
        out[f"{prefix}_Asset_Balanced_Mean_Net_Return_40D_Pct"] = float(np.mean(asset_means)) if asset_means else np.nan
        out[f"{prefix}_Positive_Asset_Rate_40D_Pct"] = float(np.mean(np.asarray(asset_means) > 0) * 100) if asset_means else np.nan
    else:
        out[f"{prefix}_Assets_40D"] = 0
        out[f"{prefix}_Asset_Balanced_Mean_Net_Return_40D_Pct"] = np.nan
        out[f"{prefix}_Positive_Asset_Rate_40D_Pct"] = np.nan
    return out


def evaluate_mask(mask: pd.Series, asset: str, close: pd.Series, split_date: pd.Timestamp) -> tuple[list[dict], list[dict]]:
    events = []
    for pos in event_positions(mask):
        if pos >= len(close):
            continue
        event_date = close.index[pos]
        record = {"Asset": asset, "Date": event_date.strftime("%Y-%m-%d")}
        for h in HORIZONS:
            if pos + h < len(close):
                record[f"R{h}"] = (float(close.iloc[pos + h]) / float(close.iloc[pos]) - 1.0) * 100.0 - ROUND_TRIP_COST_PCT
            else:
                record[f"R{h}"] = np.nan
        events.append(record)
    train = [e for e in events if pd.Timestamp(e["Date"]) <= split_date]
    test = [e for e in events if pd.Timestamp(e["Date"]) > split_date]
    return train, test


def collect_metrics(per_asset_events: dict, split: str) -> dict:
    events = [e for vals in per_asset_events.values() for e in vals[split]]
    return summarize_events(events, "Train" if split == "train" else "Test")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Walk-forward tune VFI daily + weekly Volume Divergence early-cycle parameters.")
    parser.add_argument("--output-dir", default="reports/example_parameter_tuning")
    parser.add_argument("--symbols", default=",".join(ASSETS))
    args = parser.parse_args()
    requested = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    unknown = [s for s in requested if s not in ASSETS]
    if unknown:
        parser.error(f"Unknown example asset keys: {unknown}. Valid keys: {list(ASSETS)}")

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    service = MarketDataService()
    daily_data, div_data, closes, split_dates = {}, {}, {}, {}
    coverage, errors = [], []

    for key in requested:
        ticker = ASSETS[key]
        print(f"[{key}] downloading {ticker}", flush=True)
        try:
            raw = service.get_daily_data(ticker, period="max")
            daily = completed_daily(raw)
            if daily.empty:
                raise ValueError("No completed daily OHLCV data")
            weekly = completed_weekly(daily)
            split_dates[key] = pd.Timestamp(daily.index[int(len(daily) * 0.70)]).tz_localize(None) if pd.Timestamp(daily.index[int(len(daily) * 0.70)]).tzinfo else pd.Timestamp(daily.index[int(len(daily) * 0.70)])
            closes[key] = daily["Close"].astype(float)
            daily_data[key] = daily
            if len(daily) < MIN_DAILY_BARS or len(weekly) < MIN_WEEKLY_BARS:
                coverage.append({"Asset": key, "Ticker": ticker, "Daily_Bars": len(daily), "Weekly_Bars": len(weekly), "Split_Date": str(split_dates[key].date()), "Status": "INSUFFICIENT_HISTORY_FOR_FULL_COMBINED_TEST"})
                print(f"[{key}] insufficient bars: daily={len(daily)} weekly={len(weekly)}", flush=True)
                continue
            div = divergence_features(weekly)
            # Only completed weekly oscillator values are made available to daily bars.
            aligned = div.reindex(daily.index, method="ffill")
            div_data[key] = aligned
            coverage.append({"Asset": key, "Ticker": ticker, "Daily_Bars": len(daily), "Weekly_Bars": len(weekly), "Split_Date": str(split_dates[key].date()), "Status": "READY"})
        except Exception as exc:
            errors.append({"Asset": key, "Ticker": ticker, "Error": repr(exc)})
            coverage.append({"Asset": key, "Ticker": ticker, "Daily_Bars": 0, "Weekly_Bars": 0, "Split_Date": "", "Status": "ERROR"})
            print(f"[{key}] ERROR: {exc!r}", flush=True)

    pd.DataFrame(coverage).to_csv(out / "asset_data_coverage.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out / "data_errors.csv", index=False)

    # First select VFI formula variants on training history only, with a fixed baseline setup.
    formulas = [
        {"length": length, "coef": coef, "vcoef": vcoef}
        for length, coef, vcoef in itertools.product(
            VFI_FORMULA_GRID["length"], VFI_FORMULA_GRID["coef"], VFI_FORMULA_GRID["vcoef"]
        )
    ]
    baseline_vfi_thresholds = {"vfi_near_zero_z": 0.75, "vfi_slope_min": 0.20, "vfi_max_days_to_zero": 20.0}
    vfi_formula_rows = []
    feature_cache = {}
    for formula in formulas:
        per_asset = {}
        for key in requested:
            if key not in daily_data:
                continue
            try:
                vf = vfi_features(daily_data[key], formula)
                feature_cache[(key, formula["length"], formula["coef"], formula["vcoef"])] = vf
                if key not in div_data:
                    continue
                mask = vfi_setup(vf, baseline_vfi_thresholds)
                train, test = evaluate_mask(mask, key, closes[key], split_dates[key])
                per_asset[key] = {"train": train, "test": test}
            except Exception as exc:
                errors.append({"Asset": key, "Ticker": ASSETS[key], "Error": f"VFI formula {formula}: {exc!r}"})
        train_metrics = collect_metrics(per_asset, "train")
        test_metrics = collect_metrics(per_asset, "test")
        row = {**formula, **train_metrics, **test_metrics}
        vfi_formula_rows.append(row)

    formula_rank = pd.DataFrame(vfi_formula_rows)
    if formula_rank.empty:
        raise RuntimeError("No VFI formula sensitivity results generated.")
    formula_rank = formula_rank.sort_values(
        ["Train_Asset_Balanced_Mean_Net_Return_40D_Pct", "Train_Win_Rate_40D_Pct", "Train_Assets_40D"],
        ascending=[False, False, False], na_position="last"
    ).reset_index(drop=True)
    formula_rank.insert(0, "Train_Rank", np.arange(1, len(formula_rank) + 1))
    formula_rank.to_csv(out / "vfi_formula_sensitivity.csv", index=False)
    eligible_formulas = formula_rank[
        (formula_rank["Train_Events_40D"] >= 8) & (formula_rank["Train_Assets_40D"] >= 3)
    ]
    top_formulas = eligible_formulas.head(5) if not eligible_formulas.empty else formula_rank.head(5)
    chosen_formulas = [
        {"length": int(r["length"]), "coef": float(r["coef"]), "vcoef": float(r["vcoef"])}
        for _, r in top_formulas.iterrows()
    ]
    print("Top VFI formulas selected from training data:")
    print(top_formulas[["Train_Rank", "length", "coef", "vcoef", "Train_Events_40D", "Train_Asset_Balanced_Mean_Net_Return_40D_Pct", "Test_Asset_Balanced_Mean_Net_Return_40D_Pct"]].to_string(index=False), flush=True)

    # Grid-search the cycle thresholds for VFI + weekly oscillator, using only the
    # selected VFI formula candidates. Oscillator formula/pivots remain TradingView defaults.
    threshold_sets = []
    for v_z, v_slope, v_days, d_z, d_slope, d_weeks in itertools.product(
        VFI_THRESHOLD_GRID["vfi_near_zero_z"],
        VFI_THRESHOLD_GRID["vfi_slope_min"],
        VFI_THRESHOLD_GRID["vfi_max_days_to_zero"],
        DIV_THRESHOLD_GRID["div_near_zero_z"],
        DIV_THRESHOLD_GRID["div_slope_min"],
        DIV_THRESHOLD_GRID["div_max_weeks_to_zero"],
    ):
        threshold_sets.append({
            "vfi_near_zero_z": v_z,
            "vfi_slope_min": v_slope,
            "vfi_max_days_to_zero": v_days,
            "div_near_zero_z": d_z,
            "div_slope_min": d_slope,
            "div_max_weeks_to_zero": d_weeks,
        })

    combined_rows = []
    for formula_index, formula in enumerate(chosen_formulas, start=1):
        print(f"Combined threshold grid: VFI formula {formula_index}/{len(chosen_formulas)} {formula}", flush=True)
        for params in threshold_sets:
            per_asset = {}
            for key in requested:
                cache_key = (key, formula["length"], formula["coef"], formula["vcoef"])
                if cache_key not in feature_cache or key not in div_data:
                    continue
                vf = feature_cache[cache_key]
                div = div_data[key]
                mask = vfi_setup(vf, params) & divergence_setup(div, params)
                train, test = evaluate_mask(mask, key, closes[key], split_dates[key])
                per_asset[key] = {"train": train, "test": test}
            train_metrics = collect_metrics(per_asset, "train")
            test_metrics = collect_metrics(per_asset, "test")
            combined_rows.append({**formula, **params, **train_metrics, **test_metrics})

    combined = pd.DataFrame(combined_rows)
    if combined.empty:
        raise RuntimeError("No combined parameter sensitivity results generated.")
    combined = combined.sort_values(
        ["Train_Asset_Balanced_Mean_Net_Return_40D_Pct", "Train_Win_Rate_40D_Pct", "Train_Assets_40D", "Train_Events_40D"],
        ascending=[False, False, False, False], na_position="last"
    ).reset_index(drop=True)
    combined.insert(0, "Train_Rank", np.arange(1, len(combined) + 1))
    combined.to_csv(out / "combined_parameter_sensitivity.csv", index=False)
    eligible = combined[(combined["Train_Events_40D"] >= 8) & (combined["Train_Assets_40D"] >= 3)]
    finalists = eligible.head(20) if not eligible.empty else combined.head(20)
    finalists.to_csv(out / "top_parameter_sets.csv", index=False)

    metadata = {
        "requested_assets": requested,
        "asset_tickers": {k: ASSETS[k] for k in requested},
        "timeframes": {"VFI": "daily, completed candles only", "Volume_Divergence_by_MM": "weekly, completed candles only"},
        "intraday_used": False,
        "forward_horizons_trading_days": list(HORIZONS),
        "holdout_split": "first 70% of each asset's available daily history for parameter selection; final 30% is untouched holdout",
        "round_trip_cost_assumption_pct": ROUND_TRIP_COST_PCT,
        "vfi_formula_grid": VFI_FORMULA_GRID,
        "vfi_threshold_grid": VFI_THRESHOLD_GRID,
        "divergence_threshold_grid": DIV_THRESHOLD_GRID,
        "divergence_indicator_parameters": {"vl1": 5, "vl2": 8, "derived_wmas": [13, 21, 34], "pivot_left": 5, "pivot_right": 5, "range_lower": 5, "range_upper": 60},
        "signal_definition": "Both oscillators must remain below zero, within the selected normalized distance from zero, have positive slopes on all selected lookbacks, and project a zero crossing within the selected horizon. Signals are first entry into the setup plus a 20-trading-day cooldown.",
        "selection_method": "Choose VFI formula candidates on training data only, then tune combined thresholds on training data only; report final 30% holdout metrics separately.",
        "limitations": "Small example universe and parameter search can overfit. Holdout is a historical test, not a profit guarantee. Signals enter at the signal day's close; returns subtract a 0.25 percentage-point round-trip cost.",
        "errors": errors,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("\nTOP COMBINED PARAMETER SETS (ranked on training data; holdout shown separately):")
    cols = ["Train_Rank", "length", "coef", "vcoef", "vfi_near_zero_z", "vfi_slope_min", "vfi_max_days_to_zero", "div_near_zero_z", "div_slope_min", "div_max_weeks_to_zero", "Train_Events_40D", "Train_Assets_40D", "Train_Asset_Balanced_Mean_Net_Return_40D_Pct", "Test_Events_40D", "Test_Assets_40D", "Test_Asset_Balanced_Mean_Net_Return_40D_Pct", "Test_Win_Rate_40D_Pct"]
    print(finalists[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
